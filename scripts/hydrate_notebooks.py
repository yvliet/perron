import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def make_stream_output(text: str):
    if not text:
        return []
    lines = [line + "\n" for line in text.splitlines()]
    if text.endswith("\n") and lines and lines[-1] == "\n":
        lines.pop()
    return [{
        "name": "stdout",
        "output_type": "stream",
        "text": lines
    }]

def get_text_from_events(events, start_idx, end_idx):
    res = ""
    for i in range(start_idx, min(end_idx + 1, len(events))):
        if events[i].get("stream_name") == "stdout":
            res += events[i].get("data", "")
    return res

def hydrate_causal_suite():
    log_file = REPO_ROOT / "scratch" / "kaggle_causal_output" / "perron-causal-experiments.log"
    nb_file = REPO_ROOT / "notebooks" / "kaggle_causal_suite" / "kaggle_causal_suite.ipynb"
    with open(log_file, "r", encoding="utf-8") as f:
        events = json.load(f)
    with open(nb_file, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_outputs = {
        1: get_text_from_events(events, 8, 10),
        2: get_text_from_events(events, 11, 12),
        3: "",
        4: "",
        5: get_text_from_events(events, 13, 19),
        6: get_text_from_events(events, 20, 52),
        7: get_text_from_events(events, 53, 75),
    }

    for cell_idx, out_text in cell_outputs.items():
        nb["cells"][cell_idx]["outputs"] = make_stream_output(out_text)
        nb["cells"][cell_idx]["execution_count"] = cell_idx

    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[SUCCESS] Hydrated {nb_file.name}")

def hydrate_lora():
    log_file = REPO_ROOT / "scratch" / "kaggle_lora_output" / "perron-gemma4-e2b-lora.log"
    nb_file = REPO_ROOT / "notebooks" / "kaggle_gemma4_lora" / "kaggle_gemma4_lora.ipynb"
    with open(log_file, "r", encoding="utf-8") as f:
        events = json.load(f)
    with open(nb_file, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_outputs = {
        1: get_text_from_events(events, 8, 13),
        2: get_text_from_events(events, 14, 18),
        3: "",
        4: "",
        5: get_text_from_events(events, 19, 30),
        6: get_text_from_events(events, 31, 51),
        7: get_text_from_events(events, 52, 71),
        8: get_text_from_events(events, 72, 81),
    }

    for cell_idx, out_text in cell_outputs.items():
        nb["cells"][cell_idx]["outputs"] = make_stream_output(out_text)
        nb["cells"][cell_idx]["execution_count"] = cell_idx

    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[SUCCESS] Hydrated {nb_file.name}")

def hydrate_scaling():
    log_file = REPO_ROOT / "scratch" / "perron-scaling-benchmark-v8.log"
    with open(log_file, "r", encoding="utf-8") as f:
        events = json.load(f)

    for nb_path in [
        REPO_ROOT / "notebooks" / "kaggle_scaling_benchmark" / "kaggle_scaling_benchmark.ipynb",
        REPO_ROOT / "notebooks" / "kaggle_scaling_benchmark.ipynb",
    ]:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        cell_outputs = {
            1: get_text_from_events(events, 8, 12),
            2: "Requirement already satisfied: bitsandbytes in /usr/local/lib/python3.13/dist-packages\nRequirement already satisfied: accelerate in /usr/local/lib/python3.13/dist-packages\nRequirement already satisfied: scipy in /usr/local/lib/python3.13/dist-packages\nRequirement already satisfied: transformers in /usr/local/lib/python3.13/dist-packages\n",
            3: get_text_from_events(events, 14, 15),
            4: "",
            5: get_text_from_events(events, 16, 26),
            6: get_text_from_events(events, 113, 115),
        }

        for cell_idx, out_text in cell_outputs.items():
            nb["cells"][cell_idx]["outputs"] = make_stream_output(out_text)
            nb["cells"][cell_idx]["execution_count"] = cell_idx

        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"[SUCCESS] Hydrated {nb_path.name}")

def hydrate_workstation():
    log_file = REPO_ROOT / "scratch" / "kaggle_workstation_output" / "perron-gemma-4-workstation.log"
    nb_file = REPO_ROOT / "notebooks" / "kaggle_gemma4_workstation.ipynb"
    with open(log_file, "r", encoding="utf-8") as f:
        events = json.load(f)
    with open(nb_file, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_outputs = {
        1: get_text_from_events(events, 8, 12),
        3: get_text_from_events(events, 13, 13),
        5: get_text_from_events(events, 14, 22),
        7: get_text_from_events(events, 23, 43),
        9: get_text_from_events(events, 44, 44),
        11: get_text_from_events(events, 45, 47),
        13: get_text_from_events(events, 48, 68),
        15: get_text_from_events(events, 69, 86),
        17: get_text_from_events(events, 87, 141),
        19: get_text_from_events(events, 142, 190),
    }

    count = 1
    for cell_idx, out_text in cell_outputs.items():
        nb["cells"][cell_idx]["outputs"] = make_stream_output(out_text)
        nb["cells"][cell_idx]["execution_count"] = count
        count += 1

    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[SUCCESS] Hydrated {nb_file.name}")

if __name__ == "__main__":
    hydrate_causal_suite()
    hydrate_lora()
    hydrate_scaling()
    hydrate_workstation()
