"""
Perron System Feasibility & Hardware Capability Probe (T3.1).

Probes:
1. CPU (model, cores, architecture)
2. RAM (total, available, used)
3. GPU / VRAM (CUDA, device name, total & free VRAM)
4. Free Disk (total, free space on active drive)
5. Docker Status (binary presence, daemon connectivity)
6. SWE-bench Harness (importability and evaluation runner)
7. Gemma 4 Model Loading & 20-token generation capability

Outputs raw diagnostic telemetry and outputs FEASIBILITY.md.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_cpu() -> Dict[str, Any]:
    """Inspect CPU architecture, processor model, and core counts."""
    import psutil

    proc_name = platform.processor() or "Unknown"
    logical_cores = os.cpu_count() or 0
    physical_cores = psutil.cpu_count(logical=False) or logical_cores

    return {
        "processor": proc_name,
        "machine": platform.machine(),
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "cpu_percent": psutil.cpu_percent(interval=0.1),
    }


def check_ram() -> Dict[str, Any]:
    """Inspect total, available, and used system RAM."""
    import psutil

    vmem = psutil.virtual_memory()
    return {
        "total_gb": round(vmem.total / (1024**3), 2),
        "available_gb": round(vmem.available / (1024**3), 2),
        "used_gb": round(vmem.used / (1024**3), 2),
        "percent_used": vmem.percent,
    }


def check_gpu() -> Dict[str, Any]:
    """Inspect GPU hardware and VRAM availability via PyTorch and nvidia-smi."""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": "None",
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "nvidia_smi_detected": False,
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            mem_free, mem_total = torch.cuda.mem_get_info()
            info["total_vram_gb"] = round(mem_total / (1024**3), 2)
            info["free_vram_gb"] = round(mem_free / (1024**3), 2)
    except Exception as ex:
        info["torch_cuda_error"] = str(ex)

    # Check nvidia-smi fallback
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        if smi_out:
            info["nvidia_smi_detected"] = True
            parts = [p.strip() for p in smi_out.split("\n")[0].split(",")]
            if len(parts) >= 3:
                info["smi_device_name"] = parts[0]
                info["smi_total_vram_mb"] = float(parts[1])
                info["smi_free_vram_mb"] = float(parts[2])
    except Exception:
        pass

    return info


def check_disk() -> Dict[str, Any]:
    """Inspect free and total disk space on the active workspace drive."""
    total, used, free = shutil.disk_usage(str(PROJECT_ROOT))
    return {
        "drive": str(PROJECT_ROOT.anchor),
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
    }


def check_docker() -> Dict[str, Any]:
    """Inspect Docker binary presence and daemon connectivity."""
    info: Dict[str, Any] = {
        "binary_found": False,
        "version": None,
        "daemon_running": False,
        "error": None,
    }

    docker_path = shutil.which("docker")
    if docker_path:
        info["binary_found"] = True
        info["binary_path"] = docker_path
        try:
            ver_out = subprocess.check_output(
                ["docker", "--version"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            ).strip()
            info["version"] = ver_out
        except Exception as e:
            info["version_error"] = str(e)

        try:
            info_out = subprocess.check_output(
                ["docker", "info"],
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
            ).strip()
            info["daemon_running"] = True
        except subprocess.CalledProcessError as e:
            info["daemon_running"] = False
            info["error"] = f"Docker daemon not running: {e.output[:200]}"
        except subprocess.TimeoutExpired:
            info["daemon_running"] = False
            info["error"] = "Docker command timed out after 5s"
        except Exception as e:
            info["daemon_running"] = False
            info["error"] = str(e)
    else:
        info["error"] = "docker executable not found in PATH"

    return info


def check_swebench() -> Dict[str, Any]:
    """Inspect whether the SWE-bench evaluation harness imports and is usable."""
    info: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "harness_run_evaluation": False,
        "error": None,
    }

    try:
        import swebench
        info["installed"] = True
        info["version"] = getattr(swebench, "__version__", "unknown")
        try:
            from swebench.harness.run_evaluation import run_instances
            info["harness_run_evaluation"] = True
        except Exception as ex:
            info["harness_run_evaluation"] = False
            info["harness_import_error"] = str(ex)
    except ImportError as e:
        info["installed"] = False
        info["error"] = str(e)
    except Exception as e:
        info["installed"] = False
        info["error"] = str(e)

    return info


def check_gemma_model() -> Dict[str, Any]:
    """
    Inspect whether the chosen Gemma 4 model can load and generate 20 tokens.
    """
    info: Dict[str, Any] = {
        "transformers_available": False,
        "model_loaded": False,
        "generated_20_tokens": False,
        "tokens_generated": 0,
        "sample_output": None,
        "error": None,
    }

    try:
        import transformers
        import torch
        info["transformers_available"] = True
        info["transformers_version"] = transformers.__version__
        info["torch_version"] = torch.__version__
    except ImportError as e:
        info["error"] = f"Missing dependency: {e}"
        return info

    # Check local offline paths or model configurations
    candidates = [
        "google/gemma-4-e2b-it",
        "google/gemma-4-e4b-it",
        "google/gemma-2-2b-it",
    ]

    # Attempt to load tokenizer and model
    loaded = False
    for candidate in candidates:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"Testing model loading for '{candidate}' (local_files_only=True)...")
            tok = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                candidate,
                local_files_only=True,
                torch_dtype=torch.float32,
            )
            print(f"Successfully loaded '{candidate}' locally.")
            loaded = True
            info["candidate_used"] = candidate
            info["model_loaded"] = True

            # Generate 20 tokens
            prompt = "def fibonacci(n):"
            inputs = tok(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            gen_text = tok.decode(outputs[0], skip_special_tokens=True)

            info["generated_20_tokens"] = True
            info["tokens_generated"] = len(outputs[0]) - len(inputs["input_ids"][0])
            info["sample_output"] = gen_text
            break
        except Exception as ex:
            info[f"candidate_{candidate}_error"] = str(ex)

    if not loaded:
        info["error"] = (
            "No Gemma model checkpoint found locally in offline cache. "
            "Downloading 4B+ weights requires external network access, credentials, "
            "and ~8-16 GB disk/RAM."
        )

    return info


def run_all_checks() -> Tuple[Dict[str, Any], bool, str]:
    """
    Run complete feasibility suite and return (results_dict, can_run_tests, summary_text).
    """
    print("=" * 70)
    print("RUNNING PERRON SYSTEM FEASIBILITY CHECK (T3.1)")
    print("=" * 70)

    t0 = time.time()
    cpu_info = check_cpu()
    print(f"[1/7] CPU: {cpu_info['processor']} ({cpu_info['logical_cores']} logical cores)")

    ram_info = check_ram()
    print(f"[2/7] RAM: {ram_info['total_gb']} GB total ({ram_info['available_gb']} GB available)")

    gpu_info = check_gpu()
    if gpu_info.get("cuda_available"):
        print(f"[3/7] GPU: {gpu_info['device_name']} ({gpu_info['total_vram_gb']} GB VRAM)")
    elif gpu_info.get("nvidia_smi_detected"):
        print(f"[3/7] GPU: {gpu_info.get('smi_device_name')} via nvidia-smi ({gpu_info.get('smi_total_vram_mb')} MB VRAM, CUDA torch not initialized)")
    else:
        print("[3/7] GPU: None detected / CPU-only")

    disk_info = check_disk()
    print(f"[4/7] Disk: {disk_info['free_gb']} GB free on drive {disk_info['drive']}")

    docker_info = check_docker()
    docker_status = "RUNNING" if docker_info["daemon_running"] else ("BINARY PRESENT, DAEMON STOPPED" if docker_info["binary_found"] else "NOT FOUND")
    print(f"[5/7] Docker: {docker_status}")

    swebench_info = check_swebench()
    swebench_status = f"INSTALLED (v{swebench_info['version']})" if swebench_info["installed"] else f"NOT INSTALLED ({swebench_info['error']})"
    print(f"[6/7] SWE-bench: {swebench_status}")

    gemma_info = check_gemma_model()
    gemma_status = "LOADED & GENERATED" if gemma_info["generated_20_tokens"] else f"UNAVAILABLE ({gemma_info.get('error', 'error')[:60]}...)"
    print(f"[7/7] Gemma Model: {gemma_status}")

    # Determine "can_run_tests"
    # SWE-bench Lite requires Docker daemon to execute repository unit tests inside isolated containers.
    can_run_tests = bool(docker_info["daemon_running"] and swebench_info["harness_run_evaluation"])

    results = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - t0, 2),
        "cpu": cpu_info,
        "ram": ram_info,
        "gpu": gpu_info,
        "disk": disk_info,
        "docker": docker_info,
        "swebench": swebench_info,
        "gemma": gemma_info,
        "can_run_tests": can_run_tests,
    }

    # Format human-readable raw output
    raw_lines = [
        "PERRON SYSTEM FEASIBILITY CHECK (T3.1)",
        f"Timestamp: {results['timestamp_utc']}",
        f"OS / Platform: {platform.system()} {platform.release()} ({platform.version()})",
        "",
        "1. CPU:",
        f"   Processor:     {cpu_info['processor']}",
        f"   Architecture:  {cpu_info['machine']}",
        f"   Physical Cores: {cpu_info['physical_cores']}",
        f"   Logical Cores:  {cpu_info['logical_cores']}",
        f"   CPU Load:       {cpu_info['cpu_percent']}%",
        "",
        "2. RAM:",
        f"   Total Memory:  {ram_info['total_gb']} GB",
        f"   Available:     {ram_info['available_gb']} GB",
        f"   Used:          {ram_info['used_gb']} GB ({ram_info['percent_used']}%)",
        "",
        "3. GPU / VRAM:",
        f"   CUDA Available:     {gpu_info['cuda_available']}",
        f"   Device Count:       {gpu_info['device_count']}",
        f"   Device Name:        {gpu_info['device_name']}",
        f"   Total VRAM:         {gpu_info['total_vram_gb']} GB",
        f"   Free VRAM:          {gpu_info['free_vram_gb']} GB",
        f"   NVIDIA-SMI Name:    {gpu_info.get('smi_device_name', 'N/A')}",
        f"   NVIDIA-SMI Total:   {gpu_info.get('smi_total_vram_mb', 'N/A')} MB",
        "",
        "4. Free Disk:",
        f"   Drive:         {disk_info['drive']}",
        f"   Total Space:   {disk_info['total_gb']} GB",
        f"   Used Space:    {disk_info['used_gb']} GB",
        f"   Free Space:    {disk_info['free_gb']} GB",
        "",
        "5. Docker:",
        f"   Binary Found:    {docker_info['binary_found']}",
        f"   Version:         {docker_info.get('version', 'None')}",
        f"   Daemon Running:  {docker_info['daemon_running']}",
        f"   Error:           {docker_info.get('error', 'None')}",
        "",
        "6. SWE-bench Harness:",
        f"   Installed:              {swebench_info['installed']}",
        f"   Version:                {swebench_info.get('version', 'None')}",
        f"   Harness Import:         {swebench_info['harness_run_evaluation']}",
        f"   Error:                  {swebench_info.get('error', 'None')}",
        "",
        "7. Gemma 4 Model Execution:",
        f"   Transformers Installed: {gemma_info['transformers_available']}",
        f"   Model Loaded Locally:   {gemma_info['model_loaded']}",
        f"   Generated 20 Tokens:    {gemma_info['generated_20_tokens']}",
        f"   Tokens Generated:       {gemma_info['tokens_generated']}",
        f"   Diagnostic Details:     {gemma_info.get('error', 'None')}",
        "",
        "======================================================================",
        f"VERDICT: CAN RUN TESTS? -> {'YES' if can_run_tests else 'NO'}",
        "======================================================================",
        f"Reason: {'All container and harness prerequisites satisfied.' if can_run_tests else 'Docker daemon is not running and/or SWE-bench harness cannot invoke containerized test environments on this host.'}",
    ]
    raw_output = "\n".join(raw_lines)
    return results, can_run_tests, raw_output


def generate_feasibility_markdown(output_file: Path | None = None) -> Path:
    """Generate FEASIBILITY.md with the raw output and can run tests verdict."""
    if output_file is None:
        output_file = PROJECT_ROOT / "FEASIBILITY.md"

    results, can_run_tests, raw_output = run_all_checks()

    verdict_str = "YES" if can_run_tests else "NO"

    md_content = f"""# Feasibility Diagnostic Report (T3.1)

```
{raw_output}
```

## Summary Verdict

- **Can run tests**: **{verdict_str}**
- **Docker Available**: {results['docker']['daemon_running']}
- **SWE-bench Harness Available**: {results['swebench']['harness_run_evaluation']}
- **GPU VRAM Available**: {results['gpu']['total_vram_gb']} GB ({results['gpu']['device_name']})
- **Free Disk**: {results['disk']['free_gb']} GB

> **HUMAN GATE INVARIANT**:
> Do NOT proceed to T3.2 until the human writes `PROCEED: yes` at the top of this file (`FEASIBILITY.md`).
"""

    output_file.write_text(md_content, encoding="utf-8")
    print(f"\nWritten diagnostic report to: {output_file}")
    return output_file


if __name__ == "__main__":
    generate_feasibility_markdown()
