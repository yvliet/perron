#!/usr/bin/env python3
"""
scripts/compile.py - Deterministic LaTeX compilation harness.
Enforces non-destructive diagnostics, isolates precise error lines,
and leverages local Tectonic/latexmk/pdflatex compilers.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TECTONIC = REPO_ROOT / "bin" / "tectonic.exe"

def find_compiler():
    if LOCAL_TECTONIC.exists():
        return str(LOCAL_TECTONIC), "tectonic"
    if shutil.which("tectonic"):
        return "tectonic", "tectonic"
    if shutil.which("latexmk"):
        return "latexmk", "latexmk"
    if shutil.which("pdflatex"):
        return "pdflatex", "pdflatex"
    return None, None

def run_command(cmd, cwd=None):
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        return proc.returncode, proc.stdout
    except Exception as e:
        return 1, str(e)

def compile_document(target_tex="paper/main.tex"):
    target_path = Path(target_tex)
    if not target_path.is_absolute():
        target_path = (REPO_ROOT / target_path).resolve()

    if not target_path.exists():
        print(f"[FAIL] Target LaTeX file not found: {target_path}")
        return 1

    work_dir = target_path.parent
    tex_filename = target_path.name
    base_name = target_path.stem
    pdf_file = work_dir / f"{base_name}.pdf"

    compiler_bin, compiler_type = find_compiler()
    if not compiler_bin:
        print("[FAIL] No LaTeX compiler found (tectonic, latexmk, pdflatex).")
        print("Please ensure bin/tectonic.exe exists or TeX Live/MiKTeX is installed.")
        return 1

    print(f"Compiling {tex_filename} via {compiler_type} ({compiler_bin})...")

    if compiler_type == "tectonic":
        cmd = [compiler_bin, tex_filename]
        ret, out = run_command(cmd, cwd=str(work_dir))
    elif compiler_type == "latexmk":
        cmd = [
            compiler_bin,
            "-pdf",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-synctex=1",
            tex_filename
        ]
        ret, out = run_command(cmd, cwd=str(work_dir))
    else:  # pdflatex
        cmd = [compiler_bin, "-interaction=nonstopmode", "-file-line-error", tex_filename]
        ret, out = run_command(cmd, cwd=str(work_dir))
        if shutil.which("bibtex"):
            run_command(["bibtex", base_name], cwd=str(work_dir))
        run_command(cmd, cwd=str(work_dir))
        ret, out = run_command(cmd, cwd=str(work_dir))

    if ret == 0 and pdf_file.exists():
        size_kb = pdf_file.stat().st_size / 1024
        print(f"[SUCCESS] PDF built successfully -> {pdf_file} ({size_kb:.1f} KB)")
        return 0
    else:
        print(f"[FAIL] Compilation halted with exit code {ret}")
        print("\n--- Compiler Output ---")
        print(out[-1500:] if len(out) > 1500 else out)
        return ret

if __name__ == "__main__":
    tex_arg = sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex"
    sys.exit(compile_document(tex_arg))
