"""Build a PyInstaller onefile archive for this Executa (current host platform)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "0.1.2"
ENTRY = "photo-enhancer"
DIST = ROOT / "dist"
_VENV_KEY = hashlib.sha1(f"{ROOT}-{platform.system()}-{platform.machine()}".encode("utf-8")).hexdigest()[:10]
_TMP = Path(tempfile.gettempdir()) / f"anna-pyi-{ENTRY}-{_VENV_KEY}"
VENV = _TMP / "venv"
BUILD = _TMP / "build"


def platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("amd64", "x64"):
        machine = "x86_64"
    if machine == "arm64" and system == "linux":
        machine = "aarch64"
    if system == "darwin":
        return f"darwin-{machine if machine != 'aarch64' else 'arm64'}"
    if system == "windows":
        return f"windows-{machine}"
    if system == "linux":
        return f"linux-{machine}"
    raise SystemExit(f"unsupported platform: {system}-{machine}")


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def ensure_venv() -> Path:
    py = venv_python()
    cfg = VENV / "pyvenv.cfg"
    if py.exists() and not cfg.exists():
        shutil.rmtree(VENV, ignore_errors=True)
    if not py.exists():
        VENV.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=True).create(VENV)
        py = venv_python()
    env = {**os.environ, "VIRTUAL_ENV": str(VENV)}
    subprocess.check_call(
        [str(py), "-m", "pip", "install", "-q", "-U", "pip", "pyinstaller"],
        env=env,
    )
    return py


def write_manifest(staging: Path, exe_name: str) -> None:
    (staging / "manifest.json").write_text(
        json.dumps(
            {
                "name": ENTRY,
                "version": VERSION,
                "runtime": {"binary": {"entrypoint": exe_name}},
                "permissions": {"executable": [exe_name]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def archive(staging: Path, key: str) -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    if key.startswith("windows"):
        out = DIST / f"{ENTRY}-{VERSION}-{key}.zip"
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in staging.iterdir():
                zf.write(p, arcname=p.name)
        return out
    out = DIST / f"{ENTRY}-{VERSION}-{key}.tar.gz"
    if out.exists():
        out.unlink()
    with tarfile.open(out, "w:gz") as tf:
        for p in staging.iterdir():
            tf.add(p, arcname=p.name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform-key", default=None)
    args = parser.parse_args()
    key = args.platform_key or platform_key()
    exe_name = f"{ENTRY}.exe" if key.startswith("windows") else ENTRY

    py = ensure_venv()

    if BUILD.exists():
        shutil.rmtree(BUILD)
    staging = BUILD / "staging"
    staging.mkdir(parents=True)

    cmd = [
        str(py),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        ENTRY,
        "--distpath",
        str(BUILD / "pyi"),
        "--workpath",
        str(BUILD / "work"),
        "--specpath",
        str(BUILD),
        str(ROOT / "photo_enhancer_plugin.py"),
    ]
    subprocess.check_call(cmd, cwd=ROOT)

    built = BUILD / "pyi" / exe_name
    if not built.exists():
        built = BUILD / "pyi" / ENTRY
    if not built.exists():
        raise SystemExit(f"pyinstaller output missing: {built}")

    shutil.copy2(built, staging / exe_name)
    write_manifest(staging, exe_name)
    out = archive(staging, key)
    print(f"OK {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
