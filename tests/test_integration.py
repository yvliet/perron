"""
Unit tests for Turnkey Deployment, SWE-bench Ingestion,
Trajectory Compilation, and QLoRA Training Harness.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

from perron.agent import TrajectoryResult, TurnTelemetry
from perron.runner import load_agent_yaml, run_runner
from benchmarks.swebench_loader import normalize_swebench_record
from perron.telemetry.exporter import (
    trajectory_to_sft_record,
    compile_dpo_pair,
    export_sft_dataset,
    export_dpo_dataset,
)
from training.train_lora import build_loss_masked_labels, train_lora


def test_agent_yaml_schema_validity():
    """
    Verifies that the root agent.yaml exists and contains all required turnkey deployment specifications.
    """
    repo_root = Path(__file__).resolve().parent.parent
    agent_yaml_path = repo_root / "agent.yaml"
    assert agent_yaml_path.is_file(), "Root agent.yaml is missing!"

    config = load_agent_yaml(agent_yaml_path)
    assert config.get("name") == "perron-developer-agent"
    assert "entrypoint" in config
    base_model = config["model"].get("base", "")
    assert any(
        variant in base_model
        for variant in ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-26b", "gemma-4-31b", "gemma-4"]
    ), f"Unexpected model base: {base_model}"
    assert "diffusion" in config
    assert "tools" in config
    assert len(config["tools"]) >= 3


def test_swebench_record_normalization():
    """
    Verifies that raw SWE-bench records are correctly parsed and normalized.
    """
    raw_record = {
        "instance_id": "django__django-11099",
        "repo": "django/django",
        "base_commit": "abc123def456",
        "problem_statement": "ASCIIUsernameValidator allows trailing newlines in usernames.",
        "FAIL_TO_PASS": "[\"tests.auth_tests.test_validators.ASCIIUsernameValidatorTest.test_trailing_newline\"]",
        "PASS_TO_PASS": "[\"tests.auth_tests.test_validators.ASCIIUsernameValidatorTest.test_valid\"]",
        "patch": "diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py\n",
        "test_patch": "diff --git a/tests/auth_tests/test_validators.py b/tests/auth_tests/test_validators.py\n",
    }

    normalized = normalize_swebench_record(raw_record)
    assert normalized["instance_id"] == "django__django-11099"
    assert normalized["repo"] == "django/django"
    assert normalized["base_commit"] == "abc123def456"
    assert len(normalized["fail_to_pass"]) == 1
    assert "test_trailing_newline" in normalized["fail_to_pass"][0]
    assert len(normalized["pass_to_pass"]) == 1


def test_trajectory_exporter_sft_and_dpo():
    """
    Verifies that TrajectoryResult records are compiled into clean SFT and DPO training entries.
    """
    passing_traj = TrajectoryResult(
        instance_id="task_1",
        resolved=True,
        total_turns=1,
        total_duration_seconds=1.2,
        patch_applied=True,
        test_passed=True,
        applied_edits=[{
            "file": "mod.py",
            "start_line": 1,
            "end_line": 5,
            "old_str": "def calculate(a, b):\n    return a - b",
            "new_str": "def calculate(a, b):\n    return a + b",
        }],
        thinking_traces=["Identified bug in validation logic."],
    )

    failing_traj = TrajectoryResult(
        instance_id="task_1",
        resolved=False,
        total_turns=2,
        total_duration_seconds=2.4,
        patch_applied=False,
        test_passed=False,
        failure_category="Syntax Error in Edit Block",
        applied_edits=[{
            "file": "mod.py",
            "start_line": 1,
            "end_line": 5,
            "old_str": "def calculate(a, b):\n    return a - b",
            "new_str": "def calculate(a, b)\n    invalid syntax",
        }],
        thinking_traces=["Attempted regex patch but failed."],
    )

    prompt = "Fix validation bug in mod.py"

    # 1. SFT export test
    sft_record = trajectory_to_sft_record(passing_traj, prompt)
    assert sft_record is not None
    assert sft_record["instance_id"] == "task_1"
    assert len(sft_record["messages"]) == 2
    assert sft_record["messages"][0]["role"] == "user"
    assert sft_record["messages"][1]["role"] == "model"
    assert sft_record["messages"][1]["thinking"] == "Identified bug in validation logic."
    # Verbatim code deltas must be present; placeholder comments must be absent
    model_content = sft_record["messages"][1]["content"]
    assert "# existing code" not in model_content
    assert "# updated code" not in model_content
    assert "def calculate(a, b):" in model_content
    assert "return a + b" in model_content

    # Failing trajectory should be filtered out from SFT
    failing_sft = trajectory_to_sft_record(failing_traj, prompt)
    assert failing_sft is None

    # 2. DPO preference pair test
    dpo_pair = compile_dpo_pair(passing_traj, failing_traj, prompt)
    assert dpo_pair is not None
    assert dpo_pair["instance_id"] == "task_1"
    assert "return a + b" in dpo_pair["chosen"]
    assert "invalid syntax" in dpo_pair["rejected"]
    assert "# existing code" not in dpo_pair["chosen"]
    assert "# existing code" not in dpo_pair["rejected"]

    # Export to files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sft_path = tmp_path / "sft.jsonl"
        dpo_path = tmp_path / "dpo.jsonl"

        sft_count = export_sft_dataset([(passing_traj, prompt)], sft_path)
        dpo_count = export_dpo_dataset([(passing_traj, failing_traj, prompt)], dpo_path)

        assert sft_count == 1
        assert dpo_count == 1
        assert sft_path.is_file()
        assert dpo_path.is_file()


def test_loss_masking_token_boundaries():
    """
    Verifies that response-only loss masking correctly masks prompt context
    and unmasks model thinking and edit spans.
    """
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == "<start_of_turn>model\n":
                return [101]
            elif text == "<end_of_turn>":
                return [102]
            return [ord(c) for c in text]

    tokenizer = MockTokenizer()
    # Mock input_ids: [prompt_tokens, 101 (start_of_turn model), response_tokens, 102 (end_of_turn)]
    input_ids = [1, 2, 3, 4, 101, 5, 6, 7, 8, 102]
    labels = build_loss_masked_labels(input_ids, tokenizer)

    # Prompt tokens and model turn delimiter should be masked (-100)
    assert labels[0:5] == [-100, -100, -100, -100, -100]
    # Response tokens and end_of_turn should be preserved for loss computation
    assert labels[5:10] == [5, 6, 7, 8, 102]


def test_training_dry_run_and_cpu_mock_divergence():
    """
    Verifies that the QLoRA training harness validates configurations, runs deterministic
    gradient stepping on CPU mock, and proves adapter weight divergence (Delta W != 0).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "lora_weights"
        res_dir = train_lora(
            base_model_name="google/gemma-4-31b-it",
            train_data_path="dummy.jsonl",
            output_dir=str(out_dir),
            rank=16,
            alpha=32,
            cpu_mock=True,
        )

        assert res_dir.is_dir()
        assert (res_dir / "training_config.json").is_file()
        assert (res_dir / "adapter_config.json").is_file()

        adapter_cfg = json.loads((res_dir / "adapter_config.json").read_text(encoding="utf-8"))
        assert adapter_cfg["r"] == 16
        assert adapter_cfg["lora_alpha"] == 32
        assert "q_proj" in adapter_cfg["target_modules"]
        assert adapter_cfg["verification_status"] == "verified_training_step"
        # Crucial check: verify that adapter weights actually diverged from zero
        assert adapter_cfg["adapter_weight_divergence"] > 0.0


def test_compute_git_patch_captures_untracked_files():
    """
    Verifies that compute_git_patch stages untracked files via intent-to-add
    and produces unified diff headers for newly created files.
    """
    import subprocess
    from perron.patch import compute_git_patch

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True, capture_output=True)

        # Initial commit
        init_file = repo_dir / "existing.py"
        init_file.write_text("def base():\n    return 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "existing.py"], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_dir), check=True, capture_output=True)

        # Create newly created untracked file
        new_file = repo_dir / "new_helper.py"
        new_file.write_text("def helper():\n    return 42\n", encoding="utf-8")

        patch = compute_git_patch(repo_dir)
        assert len(patch) > 0, "compute_git_patch failed to capture untracked file!"
        assert "new_helper.py" in patch
        assert "new file mode" in patch


def test_tester_cwd_resolution():
    """
    Verifies that run_targeted_test resolves relative test paths relative to target_cwd.
    """
    from perron.tester import run_targeted_test

    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        test_file = test_dir / "test_simple.py"
        test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

        # Pass relative test path with explicit cwd
        res = run_targeted_test("test_simple.py", cwd=test_dir, timeout_seconds=10.0)
        # Should NOT fail with exit code 4 (file not found)
        assert res.exit_code != 4, f"Failed with file not found: {res.output}"
        assert res.passed is True


def test_runner_fail_fast_on_invalid_backend():
    """
    Verifies that create_backend_from_config raises descriptive exceptions
    instead of silently falling back to OfflineReplayBackend.
    """
    from perron.runner import create_backend_from_config

    with pytest.raises(ValueError, match="Unknown model backend type"):
        create_backend_from_config({"backend": "unsupported_backend_xyz"})


def test_runner_headless_smoke_test():
    """
    Verifies that run_runner executes headlessly using turnkey manifest configuration.
    """
    import yaml
    repo_root = Path(__file__).resolve().parent.parent
    agent_yaml_path = repo_root / "agent.yaml"
    assert agent_yaml_path.is_file(), "Root agent.yaml is missing!"

    config = load_agent_yaml(agent_yaml_path)
    assert config.get("name") == "perron-developer-agent"
    assert "diffusion" in config

    # Test runner execution with replay backend configuration
    test_cfg = dict(config)
    test_cfg["model"] = dict(config.get("model", {}))
    test_cfg["model"]["backend"] = "replay"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg_file = tmp_path / "test_agent.yaml"
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(test_cfg, f)

        sample_file = tmp_path / "sample.py"
        sample_file.write_text("def test_fn(): pass\n", encoding="utf-8")

        result = run_runner(
            repo_dir=tmp_path,
            issue_text="Verify turnkey agent runner execution",
            config_path=cfg_file,
            max_turns=1,
        )
        assert "resolved" in result
        assert "total_turns" in result
        assert result["total_turns"] >= 1
