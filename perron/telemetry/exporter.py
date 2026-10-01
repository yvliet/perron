"""
Trajectory Serialization & Dataset Compiler for SFT and DPO Training.

Transforms PerronAgent TrajectoryResult records into publication-grade training datasets:
1. ShareGPT / OpenAI SFT JSONL format with native Gemma 4 thinking tags and XML edit blocks.
2. DPO preference pairs contrasting verified passing patches (chosen) against syntax/test failures (rejected).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from perron.agent import TrajectoryResult, TurnTelemetry


def format_gemma4_turn(role: str, content: str, thinking_trace: Optional[str] = None) -> str:
    """
    Formats a single conversational turn with native Gemma 4 delimiters and thinking tokens.
    """
    if role == "model":
        parts = [f"<start_of_turn>model\n"]
        if thinking_trace:
            parts.append(f"<|think|>\n{thinking_trace.strip()}\n<|/think|>\n")
        parts.append(f"{content.strip()}<end_of_turn>")
        return "".join(parts)
    else:
        return f"<start_of_turn>{role}\n{content.strip()}<end_of_turn>"


def trajectory_to_sft_record(traj: TrajectoryResult, prompt_text: str) -> Optional[Dict[str, Any]]:
    """
    Converts a single verified TrajectoryResult into an SFT training instance.
    Enforces quality filtering: only admitted if resolved and test passed.
    """
    if not (traj.resolved and traj.test_passed and traj.patch_applied):
        return None

    # Construct conversation turns
    messages: List[Dict[str, str]] = []
    messages.append({
        "role": "user",
        "content": prompt_text.strip(),
    })

    # Model response with thinking trace
    thinking = traj.thinking_traces[0] if traj.thinking_traces else ""
    # Reconstruct edit blocks from applied edits with verbatim code deltas
    response_content = ""
    for edit in traj.applied_edits:
        file_path = edit.get("file", "unknown.py")
        s_line = edit.get("start_line", 1)
        e_line = edit.get("end_line", 1)
        old_str = edit.get("old_str", "")
        new_str = edit.get("new_str", "")
        if old_str or new_str:
            response_content += f'<edit file="{file_path}" start_line="{s_line}" end_line="{e_line}">\n<old>\n{old_str}\n</old>\n<new>\n{new_str}\n</new>\n</edit>\n'
        else:
            response_content += f'<edit file="{file_path}" start_line="{s_line}" end_line="{e_line}">\n</edit>\n'

    if not response_content:
        response_content = "<!-- Valid patch applied and verified -->"

    messages.append({
        "role": "model",
        "content": response_content.strip(),
        "thinking": thinking,
    })

    return {
        "instance_id": traj.instance_id,
        "resolved": traj.resolved,
        "turns_count": traj.total_turns,
        "messages": messages,
    }


def compile_dpo_pair(
    winning_traj: TrajectoryResult,
    losing_traj: TrajectoryResult,
    prompt_text: str,
) -> Optional[Dict[str, Any]]:
    """
    Constructs a DPO preference record contrasting a verified passing resolution
    against an execution or syntax failure on the same instance.
    """
    if not (winning_traj.resolved and winning_traj.test_passed):
        return None
    if losing_traj.resolved and losing_traj.test_passed:
        return None

    win_thinking = winning_traj.thinking_traces[0] if winning_traj.thinking_traces else ""
    lose_thinking = losing_traj.thinking_traces[0] if losing_traj.thinking_traces else ""

    chosen_edits = ""
    for edit in winning_traj.applied_edits:
        file_path = edit.get("file", "unknown.py")
        s_line = edit.get("start_line", 1)
        e_line = edit.get("end_line", 1)
        old_str = edit.get("old_str", "")
        new_str = edit.get("new_str", "")
        if old_str or new_str:
            chosen_edits += f'<edit file="{file_path}" start_line="{s_line}" end_line="{e_line}">\n<old>\n{old_str}\n</old>\n<new>\n{new_str}\n</new>\n</edit>\n'
        else:
            chosen_edits += f'<edit file="{file_path}" start_line="{s_line}" end_line="{e_line}">\n</edit>\n'
    if not chosen_edits:
        chosen_edits = f"<!-- Verified working patch for {winning_traj.instance_id} -->"

    rejected_edits = ""
    for edit in losing_traj.applied_edits:
        file_path = edit.get("file", "unknown.py")
        s_line = edit.get("start_line", 1)
        e_line = edit.get("end_line", 1)
        old_str = edit.get("old_str", "")
        new_str = edit.get("new_str", "")
        if old_str or new_str:
            rejected_edits += f'<edit file="{file_path}" start_line="{s_line}" end_line="{e_line}">\n<old>\n{old_str}\n</old>\n<new>\n{new_str}\n</new>\n</edit>\n'
        else:
            rejected_edits += f'<edit file="{file_path}" start_line="{s_line}" end_line="{e_line}">\n</edit>\n'
    if not rejected_edits:
        rejected_edits = f"<!-- Failed repair attempt: {losing_traj.failure_category or 'Test Regression'} -->"

    chosen_response = f"<|think|>\n{win_thinking}\n<|/think|>\n{chosen_edits.strip()}"
    rejected_response = f"<|think|>\n{lose_thinking}\n<|/think|>\n{rejected_edits.strip()}"

    return {
        "instance_id": winning_traj.instance_id,
        "prompt": prompt_text.strip(),
        "chosen": chosen_response,
        "rejected": rejected_response,
        "chosen_turns": winning_traj.total_turns,
        "rejected_failure": losing_traj.failure_category,
    }


def export_sft_dataset(
    trajectories: List[Tuple[TrajectoryResult, str]],
    output_path: Path,
) -> int:
    """
    Exports verified trajectories to an SFT JSONL dataset file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for traj, prompt in trajectories:
            record = trajectory_to_sft_record(traj, prompt)
            if record:
                f.write(json.dumps(record) + "\n")
                count += 1
    return count


def export_dpo_dataset(
    pairs: List[Tuple[TrajectoryResult, TrajectoryResult, str]],
    output_path: Path,
) -> int:
    """
    Exports preference pairs to a DPO JSONL dataset file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for win, lose, prompt in pairs:
            record = compile_dpo_pair(win, lose, prompt)
            if record:
                f.write(json.dumps(record) + "\n")
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Perron Trajectory SFT/DPO Dataset Compiler")
    parser.add_argument("--output-sft", type=str, default="data/trajectories_sft.jsonl", help="SFT JSONL path")
    parser.add_argument("--output-dpo", type=str, default="data/trajectories_dpo.jsonl", help="DPO JSONL path")
    args = parser.parse_args()

    sft_out = Path(args.output_sft).resolve()
    dpo_out = Path(args.output_dpo).resolve()

    print(f"Dataset compiler initialized. Target paths:")
    print(f"  SFT: {sft_out}")
    print(f"  DPO: {dpo_out}")


if __name__ == "__main__":
    main()
