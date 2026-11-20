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
VERSION = "0.1.5"
ENTRY = "photo-enhancer"
# Shim / archive identity must match the minted tool_id (Agent bin/ name).
TOOL_ID = "tool-akshay-photo-enhancer-72jdd7jk"
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


def _read_u32(data: bytes, off: int, *, be: bool = False) -> int:
    chunk = data[off : off + 4]
    return int.from_bytes(chunk, "big" if be else "little")


def detect_binary_arch(path: Path) -> str:
    data = path.read_bytes()[:64]
    if len(data) < 20:
        raise SystemExit(f"binary too small to inspect: {path}")
    magic = _read_u32(data, 0)
    if magic in (0xFEEDFACF, 0xCFFAEDFE):
        be = magic == 0xCFFAEDFE
        cpu = _read_u32(data, 4, be=be)
        if cpu == 0x0100000C:
            return "arm64"
        if cpu == 0x01000007:
            return "x86_64"
        raise SystemExit(f"unsupported Mach-O cpu 0x{cpu:08x} in {path}")
    if data[:4] == b"\x7fELF":
        endian = "little" if data[5] == 1 else "big"
        machine = int.from_bytes(data[18:20], endian)
        if machine == 62:
            return "x86_64"
        if machine == 183:
            return "aarch64"
        raise SystemExit(f"unsupported ELF machine {machine} in {path}")
    if data[:2] == b"MZ":
        pe_off = int.from_bytes(data[0x3C:0x40], "little")
        pe = path.read_bytes()[pe_off : pe_off + 6]
        if pe[:4] != b"PE\0\0":
            raise SystemExit(f"invalid PE header in {path}")
        machine = int.from_bytes(pe[4:6], "little")
        if machine == 0x8664:
            return "x86_64"
        if machine == 0xAA64:
            return "arm64"
        raise SystemExit(f"unsupported PE machine 0x{machine:04x} in {path}")
    raise SystemExit(f"unrecognized binary format: {path}")


def assert_platform_key_matches_host(key: str) -> None:
    host = platform_key()
    if key != host:
        raise SystemExit(
            f"--platform-key {key} does not match this builder ({host}). "
            "Build on a real matching runner (e.g. macos-13 for darwin-x86_64); "
            "do not relabel cross-arch output."
        )


def assert_binary_matches_key(path: Path, key: str) -> None:
    arch = detect_binary_arch(path)
    expected = key.split("-", 1)[1]
    if expected == "aarch64":
        ok = arch in ("aarch64", "arm64")
    else:
        ok = arch == expected
    if not ok:
        raise SystemExit(
            f"arch mismatch: archive key {key} but binary {path.name} is {arch}"
        )


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
                "name": TOOL_ID,
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
    assert_platform_key_matches_host(key)
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
    assert_binary_matches_key(staging / exe_name, key)
    write_manifest(staging, exe_name)
    out = archive(staging, key)
    print(f"OK {out} arch={detect_binary_arch(staging / exe_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
