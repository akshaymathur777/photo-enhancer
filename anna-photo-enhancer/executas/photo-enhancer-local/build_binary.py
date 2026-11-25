"""Build a PyInstaller onefile archive for the local classical Executa."""
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
VERSION = "0.1.6"
ENTRY = "photo-enhancer-local"
TOOL_ID = "tool-akshay-photo-enhancer-local-cwgtd7ym"
DIST = ROOT / "dist"
MODELS = ROOT / "models"
U2NETP_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
U2NETP_MD5 = "8e83ca70e441ab06c318d82300c84806"
_VENV_KEY = hashlib.sha1(f"{ROOT}-{platform.system()}-{platform.machine()}".encode("utf-8")).hexdigest()[:10]
_TMP = Path(tempfile.gettempdir()) / f"anna-pyi-{ENTRY}-{_VENV_KEY}"
VENV = _TMP / "venv"
# Keep PyInstaller worktrees off OneDrive / cross-OS mounts (parallel Windows+WSL races).
BUILD = _TMP / "build"

# Keep the freeze lean: rembg + classical deps only (no torch/transformers from host site-packages).
# Install onnxruntime explicitly instead of rembg[cpu] — under Rosetta x86_64 the
# rembg[cpu] extra can pull numba/llvmlite and try to compile LLVM from source.
DEPS = [
    "pyinstaller",
    "Pillow",
    "numpy",
    "opencv-python-headless",
    "onnxruntime",
    "rembg",
    "filetype",
]
# Native packages must come from wheels (esp. darwin-x86_64 under Rosetta).
BINARY_ONLY = [
    "llvmlite",
    "numba",
    "onnxruntime",
    "numpy",
    "opencv-python-headless",
    "Pillow",
]


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
    """Return a platform_key-compatible arch token for a built executable."""
    data = path.read_bytes()[:64]
    if len(data) < 20:
        raise SystemExit(f"binary too small to inspect: {path}")

    # Mach-O 64-bit (native endian or swapped)
    magic = _read_u32(data, 0)
    if magic in (0xFEEDFACF, 0xCFFAEDFE):
        be = magic == 0xCFFAEDFE
        cpu = _read_u32(data, 4, be=be)
        if cpu == 0x0100000C:
            return "arm64"
        if cpu == 0x01000007:
            return "x86_64"
        raise SystemExit(f"unsupported Mach-O cpu 0x{cpu:08x} in {path}")

    # ELF
    if data[:4] == b"\x7fELF":
        endian = "little" if data[5] == 1 else "big"
        machine = int.from_bytes(data[18:20], endian)
        if machine == 62:
            return "x86_64"
        if machine == 183:
            return "aarch64"
        raise SystemExit(f"unsupported ELF machine {machine} in {path}")

    # PE
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
    """Refuse to label an archive for a different CPU than this builder.

    GitHub's macos-latest + setup-python architecture=x64 previously produced
    ARM64 Mach-O bytes still labeled darwin-x86_64 — Anna review rejected that.
    """
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
    # linux arm uses aarch64 in platform keys; Mach-O/PE use arm64.
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
    subprocess.check_call([str(py), "-m", "pip", "install", "-q", "-U", "pip"], env=env)
    only_binary = ",".join(BINARY_ONLY)
    subprocess.check_call(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "-q",
            "--prefer-binary",
            f"--only-binary={only_binary}",
            *DEPS,
        ],
        env=env,
    )
    return py


def ensure_u2netp() -> Path:
    """Fetch the ~4.6MB lite rembg weights once for baking into the freeze."""
    MODELS.mkdir(parents=True, exist_ok=True)
    dest = MODELS / "u2netp.onnx"
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        digest = hashlib.md5(dest.read_bytes()).hexdigest()
        if digest == U2NETP_MD5:
            return dest
    # Prefer a machine-local rembg cache when offline CI already warmed it.
    for cached in (
        Path.home() / ".rembg" / "models" / "u2netp" / "u2netp.onnx",
        Path.home() / ".u2net" / "u2netp.onnx",
    ):
        if cached.is_file() and cached.stat().st_size > 1_000_000:
            shutil.copy2(cached, dest)
            digest = hashlib.md5(dest.read_bytes()).hexdigest()
            if digest == U2NETP_MD5:
                return dest
    import urllib.request

    print(f"Downloading u2netp -> {dest}")
    with urllib.request.urlopen(U2NETP_URL, timeout=120) as resp:
        data = resp.read()
    digest = hashlib.md5(data).hexdigest()
    if digest != U2NETP_MD5:
        raise SystemExit(f"u2netp md5 mismatch: got {digest}, want {U2NETP_MD5}")
    dest.write_bytes(data)
    return dest


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
    u2netp = ensure_u2netp()

    if BUILD.exists():
        shutil.rmtree(BUILD)
    staging = BUILD / "staging"
    staging.mkdir(parents=True)

    meta_pkgs = [
        "rembg",
        "pymatting",
        "pooch",
        "onnxruntime",
        "numpy",
        "Pillow",
        "opencv-python-headless",
        "jsonschema",
        "filetype",
    ]
    # PyInstaller --add-data uses OS path separator between src and dest-in-bundle.
    add_data = f"{u2netp}{os.pathsep}models"
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
        "--hidden-import",
        "classical_ops",
        "--collect-all",
        "rembg",
        "--collect-all",
        "onnxruntime",
        "--add-data",
        add_data,
        "--exclude-module",
        "torch",
        "--exclude-module",
        "torchvision",
        "--exclude-module",
        "transformers",
        "--exclude-module",
        "tensorflow",
        "--exclude-module",
        "sklearn",
        "--exclude-module",
        "pandas",
    ]
    for pkg in meta_pkgs:
        cmd.extend(["--copy-metadata", pkg])
    cmd.append(str(ROOT / "photo_enhancer_local_plugin.py"))
    subprocess.check_call(cmd, cwd=ROOT)

    built = BUILD / "pyi" / exe_name
    if not built.exists():
        built = BUILD / "pyi" / ENTRY
    if not built.exists():
        raise SystemExit(f"pyinstaller output missing under {BUILD / 'pyi'}")

    shutil.copy2(built, staging / exe_name)
    assert_binary_matches_key(staging / exe_name, key)
    # Also ship flat weights next to the entrypoint for non-_MEIPASS installs.
    shutil.copy2(u2netp, staging / "u2netp.onnx")
    write_manifest(staging, exe_name)
    out = archive(staging, key)
    print(f"OK {out} arch={detect_binary_arch(staging / exe_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
