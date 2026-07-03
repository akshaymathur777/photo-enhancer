import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"
VENV_PYTHON = BACKEND / "venv" / "Scripts" / "python.exe"
PORT = 8000
URL = f"http://127.0.0.1:{PORT}"


def ensure_frontend_built() -> None:
    if DIST.exists() and (DIST / "index.html").exists():
        return
    print("Building GUI (first run only)...")
    subprocess.run(["npm", "install"], cwd=FRONTEND, shell=True, check=True)
    subprocess.run(["npm", "run", "build"], cwd=FRONTEND, shell=True, check=True)
    print("GUI built.")


def main() -> None:
    if not VENV_PYTHON.exists():
        print("Virtual environment not found. Run setup first:")
        print("  cd backend")
        print("  python -m venv venv")
        print("  venv\\Scripts\\pip install -r requirements.txt")
        sys.exit(1)

    ensure_frontend_built()

    print(f"Starting Photo Enhancer at {URL}")
    proc = subprocess.Popen(
        [str(VENV_PYTHON), "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=BACKEND,
    )

    for _ in range(30):
        try:
            import urllib.request

            urllib.request.urlopen(f"{URL}/api/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        print("Server failed to start.")
        proc.terminate()
        sys.exit(1)

    webbrowser.open(URL)
    print("Press Ctrl+C to quit.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nStopped.")


if __name__ == "__main__":
    main()
