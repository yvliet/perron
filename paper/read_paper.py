#!/usr/bin/env python3
"""
paper/read_paper.py - Local Academic Paper Reader and Compilation Watcher.

Usage:
    python paper/read_paper.py             # Compiles if needed and opens main.pdf
    python paper/read_paper.py --compile   # Forces fresh compile and opens PDF
    python paper/read_paper.py --watch     # Live watcher: auto-recompiles on edit
    python paper/read_paper.py --server    # Local HTTP live-reload viewer (localhost:8080)
"""

import sys
import os
import time
import argparse
import webbrowser
import subprocess
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT / "paper"
MAIN_TEX = PAPER_DIR / "main.tex"
MAIN_PDF = PAPER_DIR / "main.pdf"
COMPILE_SCRIPT = REPO_ROOT / "scripts" / "compile.py"
CHROME_PATH = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

def compile_pdf():
    print("[1/2] Compiling paper/main.tex...")
    cmd = [sys.executable, str(COMPILE_SCRIPT), str(MAIN_TEX)]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return res.returncode == 0

def open_pdf_viewer(target_pdf=MAIN_PDF):
    if not target_pdf.exists():
        print(f"[FAIL] Target PDF not found: {target_pdf}")
        return False

    pdf_uri = target_pdf.resolve().as_uri()
    print(f"[2/2] Opening PDF reader -> {target_pdf.name}")

    # Prefer Google Chrome if available for high-fidelity tabbed viewing
    if CHROME_PATH.exists():
        try:
            subprocess.Popen([str(CHROME_PATH), pdf_uri])
            print("  Opened in Google Chrome reader.")
            return True
        except Exception:
            pass

    # Fallback to system default PDF reader (Edge, Sumatra, Acrobat)
    try:
        if os.name == 'nt':
            os.startfile(str(target_pdf))
        else:
            webbrowser.open(pdf_uri)
        print("  Opened in system default reader.")
        return True
    except Exception as e:
        print(f"  Notice: Could not automatically launch viewer ({e}).")
        print(f"  You can open: {target_pdf}")
        return False

def get_watched_timestamps():
    timestamps = {}
    watch_dirs = [PAPER_DIR, PAPER_DIR / "figures"]
    for d in watch_dirs:
        if d.exists():
            for f in d.glob("*"):
                if f.is_file() and f.suffix.lower() in {".tex", ".bib", ".png", ".pdf", ".py", ".json"}:
                    try:
                        timestamps[str(f)] = f.stat().st_mtime
                    except Exception:
                        pass
    return timestamps

def run_watch_loop():
    print("============================================================")
    print("STARTING PERRON LIVE LATEX COMPILATION WATCHER")
    print("Watching paper/*.tex, paper/*.bib, paper/figures/*...")
    print("Press Ctrl+C to terminate.")
    print("============================================================")

    last_state = get_watched_timestamps()
    while True:
        try:
            time.sleep(1.0)
            current_state = get_watched_timestamps()
            changed = []
            for path, mtime in current_state.items():
                if path not in last_state or mtime > last_state[path]:
                    if not path.endswith("main.pdf"):
                        changed.append(os.path.basename(path))

            if changed:
                timestamp_str = time.strftime("%H:%M:%S")
                print(f"\n[{timestamp_str}] Detected change in: {', '.join(changed)}")
                compile_pdf()
                last_state = current_state
        except KeyboardInterrupt:
            print("\nWatcher stopped.")
            break

class PaperServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def do_GET(self):
        if self.path in {"/", "/index.html", "/reader"}:
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Perron Academic Reader</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    header {{
      background: #1e293b;
      border-bottom: 1px solid #334155;
      padding: 10px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .title {{ font-size: 14px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
    .badge {{
      background: #0369a1; color: #e0f2fe; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600;
    }}
    .actions {{ display: flex; gap: 10px; align-items: center; }}
    button, a.btn {{
      background: #2563eb; color: #ffffff; text-decoration: none; padding: 6px 14px; border-radius: 6px;
      font-size: 12px; font-weight: 600; border: none; cursor: pointer; display: inline-flex; align-items: center; gap: 4px;
    }}
    button:hover, a.btn:hover {{ background: #1d4ed8; }}
    .container {{ flex: 1; position: relative; }}
    iframe {{ width: 100%; height: 100%; border: none; }}
  </style>
</head>
<body>
  <header>
    <div class="title">
      <span>Perron: Local Academic Paper Reader</span>
      <span class="badge">Tectonic Native</span>
    </div>
    <div class="actions">
      <span style="font-size: 12px; color: #94a3b8;">12 Pages · Conference Format</span>
      <button onclick="reloadViewer()">↻ Reload PDF</button>
      <a class="btn" href="/paper/main.pdf" download>Download PDF</a>
    </div>
  </header>
  <div class="container">
    <iframe id="pdfFrame" src="/paper/main.pdf#toolbar=1&navpanes=1"></iframe>
  </div>
  <script>
    function reloadViewer() {{
      const frame = document.getElementById('pdfFrame');
      frame.src = '/paper/main.pdf?t=' + new Date().getTime() + '#toolbar=1&navpanes=1';
    }}
    // Auto-poll for updates every 3 seconds
    let lastModified = 0;
    setInterval(async () => {{
      try {{
        const res = await fetch('/paper/main.pdf', {{ method: 'HEAD' }});
        const lm = res.headers.get('Last-Modified');
        if (lastModified && lm && lm !== lastModified) {{
          reloadViewer();
        }}
        lastModified = lm;
      }} catch (e) {{}}
    }}, 3000);
  </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
        else:
            super().do_GET()

def run_server(port=8080):
    server = HTTPServer(("localhost", port), PaperServerHandler)
    url = f"http://localhost:{port}/"
    print(f"[SERVER] Starting local Academic Paper Reader on {url}")
    threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer terminated.")

def main():
    parser = argparse.ArgumentParser(description="Read and compile Perron paper locally.")
    parser.add_argument("--compile", action="store_true", help="Force fresh compilation before opening")
    parser.add_argument("--watch", action="store_true", help="Watch for source file edits and auto-recompile")
    parser.add_argument("--server", action="store_true", help="Launch live-reload browser reader at localhost:8080")
    parser.add_argument("--port", type=int, default=8080, help="Port for local reader server (default: 8080)")
    args = parser.parse_args()

    if args.server:
        if not MAIN_PDF.exists() or args.compile:
            compile_pdf()
        run_server(port=args.port)
        return

    if args.watch:
        if not MAIN_PDF.exists() or args.compile:
            compile_pdf()
        open_pdf_viewer(MAIN_PDF)
        run_watch_loop()
        return

    # Default action: compile if needed, then open
    needs_compile = args.compile or not MAIN_PDF.exists()
    if needs_compile:
        success = compile_pdf()
        if not success:
            sys.exit(1)
    else:
        # Check if source is newer than PDF
        pdf_mtime = MAIN_PDF.stat().st_mtime
        if MAIN_TEX.stat().st_mtime > pdf_mtime:
            compile_pdf()

    open_pdf_viewer(MAIN_PDF)

if __name__ == "__main__":
    main()
