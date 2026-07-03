"""Photo Enhancer desktop entry — native window via pywebview + local FastAPI server."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
VENV_PYTHON = BACKEND / "venv" / "Scripts" / "python.exe"

WINDOW_TITLE = "Photo Enhancer"
MIN_WIDTH = 1024
MIN_HEIGHT = 700
DEFAULT_PORT = 8000

LOADING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Photo Enhancer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: "Segoe UI", system-ui, sans-serif;
    background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .card {{
    text-align: center;
    padding: 2.5rem 3rem;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.2);
    box-shadow: 0 24px 48px rgba(0, 0, 0, 0.35);
    max-width: 420px;
    width: 90%;
  }}
  h1 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem; }}
  p {{ color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem; line-height: 1.5; }}
  .spinner {{
    width: 40px; height: 40px; margin: 0 auto 1rem;
    border: 3px solid rgba(148, 163, 184, 0.25);
    border-top-color: #6366f1;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }}
  .progress-wrap {{ display: none; margin-top: 0.5rem; }}
  .progress-wrap.active {{ display: block; }}
  .progress-bar {{
    height: 6px; border-radius: 999px;
    background: rgba(148, 163, 184, 0.2);
    overflow: hidden; margin-bottom: 0.5rem;
  }}
  .progress-fill {{
    height: 100%; width: 0%;
    background: linear-gradient(90deg, #6366f1, #818cf8);
    border-radius: 999px;
    transition: width 0.3s ease;
  }}
  .progress-label {{ font-size: 0.8rem; color: #94a3b8; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
  <div class="card">
    <h1>Photo Enhancer</h1>
    <p id="status">Starting the editor&hellip;<br><span style="font-size:0.85rem">Models download once, then work offline.</span></p>
    <div class="spinner" id="spinner" aria-hidden="true"></div>
    <div class="progress-wrap" id="progress-wrap">
      <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
      <div class="progress-label" id="progress-label"></div>
    </div>
  </div>
  <script>
    (function () {{
      var base = "{base_url}";
      var statusEl = document.getElementById("status");
      var wrap = document.getElementById("progress-wrap");
      var fill = document.getElementById("progress-fill");
      var label = document.getElementById("progress-label");

      window.__updateDownloadProgress = function (p) {{
        if (!p || p.status !== "downloading") {{
          wrap.classList.remove("active");
          return;
        }}
        wrap.classList.add("active");
        statusEl.innerHTML = "Downloading AI models&hellip;<br><span style=\\"font-size:0.85rem\\">Once complete, everything works offline.</span>";
        var pct = Math.round(p.percent || 0);
        fill.style.width = pct + "%";
        label.textContent = (p.message || p.current_model_name || "Downloading") + " — " + pct + "%";
      }};
    }})();
  </script>
</body>
</html>"""


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def _ensure_backend_on_path() -> None:
    backend_dir = BACKEND
    if _is_frozen():
        backend_dir = Path(sys._MEIPASS) / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))


def _reexec_with_venv_if_needed() -> None:
    if _is_frozen():
        return
    if not VENV_PYTHON.is_file():
        return
    if Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return
    os_exec = [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
    os.execv(str(VENV_PYTHON), os_exec)


def find_free_port(start: int = DEFAULT_PORT, attempts: int = 50) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{start + attempts - 1}")


def wait_for_server(url: str, timeout_sec: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    health_url = f"{url.rstrip('/')}/api/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.25)
    return False


def _fetch_download_progress(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/models/download-progress", timeout=2) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def wait_for_downloads(url: str, window, timeout_sec: float = 7200.0) -> None:
    """Poll download progress and update loading screen until idle or timeout."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        progress = _fetch_download_progress(url)
        if progress is None or progress.get("status") != "downloading":
            return
        try:
            window.evaluate_js(
                f"window.__updateDownloadProgress && window.__updateDownloadProgress({json.dumps(progress)})"
            )
        except Exception:
            pass
        time.sleep(0.8)


class _ServerHandle:
    def __init__(self) -> None:
        self.server = None
        self.thread: threading.Thread | None = None

    def start(self, host: str, port: int) -> None:
        import uvicorn

        from main import app  # noqa: WPS433 — backend entry after path setup

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True, name="uvicorn")
        self.thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=timeout)


def main() -> None:
    _reexec_with_venv_if_needed()
    _ensure_backend_on_path()

    from paths import configure_runtime_paths

    configure_runtime_paths()

    import webview

    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    server = _ServerHandle()
    server.start("127.0.0.1", port)

    icon = None
    try:
        from paths import get_app_icon

        icon_path = get_app_icon()
        if icon_path is not None:
            icon = str(icon_path)
    except Exception:
        pass

    window = webview.create_window(
        WINDOW_TITLE,
        html=LOADING_HTML.format(base_url=url),
        width=1280,
        height=800,
        min_size=(MIN_WIDTH, MIN_HEIGHT),
        resizable=True,
    )

    def _load_app_when_ready() -> None:
        if wait_for_server(url):
            wait_for_downloads(url, window)
            window.load_url(url)
        else:
            window.load_html(
                "<body style='font-family:Segoe UI,sans-serif;padding:2rem'>"
                "<h2>Photo Enhancer failed to start</h2>"
                "<p>The local server did not respond. Try restarting the app.</p>"
                "</body>"
            )

    threading.Thread(target=_load_app_when_ready, daemon=True, name="server-wait").start()

    try:
        start_kwargs: dict = {"gui": "edgechromium"}
        if icon:
            start_kwargs["icon"] = icon
        webview.start(**start_kwargs)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
