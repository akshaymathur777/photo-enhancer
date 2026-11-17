#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# When invoked from Windows path via WSL, prefer the mounted path.
if [[ ! -f "$ROOT/executas/photo-enhancer/dist/photo-enhancer-0.1.1-linux-x86_64.tar.gz" ]]; then
  ROOT="/mnt/c/Users/aksha/OneDrive/Documents/Projects/photo-enhancer/anna-photo-enhancer"
fi
WORKDIR="${TMPDIR:-/tmp}/anna-fresh-$$"
mkdir -p "$WORKDIR/host" "$WORKDIR/local"
trap 'rm -rf "$WORKDIR"' EXIT

tar -xzf "$ROOT/executas/photo-enhancer/dist/photo-enhancer-0.1.1-linux-x86_64.tar.gz" -C "$WORKDIR/host"
tar -xzf "$ROOT/executas/photo-enhancer-local/dist/photo-enhancer-local-0.1.1-linux-x86_64.tar.gz" -C "$WORKDIR/local"
chmod +x "$WORKDIR/host/photo-enhancer" "$WORKDIR/local/photo-enhancer-local"

cat >"$WORKDIR/host-req.jsonl" <<'EOF'
{"jsonrpc":"2.0","method":"describe","id":1}
{"jsonrpc":"2.0","method":"invoke","id":2,"params":{"tool":"ping","arguments":{}}}
EOF

echo "=== HOST describe+ping ==="
timeout 30 "$WORKDIR/host/photo-enhancer" <"$WORKDIR/host-req.jsonl"

python3 - <<'PY' >"$WORKDIR/local-req.jsonl"
import base64, json, struct, zlib

def chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

raw = b"".join(b"\x00" + bytes([255, 0, 0]) * 2 for _ in range(2))
png = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(raw))
    + chunk(b"IEND", b"")
)
b64 = base64.b64encode(png).decode()
reqs = [
    {"jsonrpc": "2.0", "method": "describe", "id": 1},
    {"jsonrpc": "2.0", "method": "invoke", "id": 2, "params": {"tool": "capabilities", "arguments": {}}},
    {
        "jsonrpc": "2.0",
        "method": "invoke",
        "id": 3,
        "params": {"tool": "local_enhance", "arguments": {"image_png_b64": b64, "strength": 1.0}},
    },
]
print("\n".join(json.dumps(r) for r in reqs))
PY

echo "=== LOCAL describe+capabilities+enhance ==="
timeout 180 "$WORKDIR/local/photo-enhancer-local" <"$WORKDIR/local-req.jsonl" | tee "$WORKDIR/local-out.jsonl" | head -c 6000
echo
python3 - <<'PY'
import json, pathlib, sys
p = pathlib.Path(__import__("os").environ.get("WORKDIR",""))
# fallback read from argv
out = pathlib.Path(sys.argv[1])
ok = 0
for line in out.read_text().splitlines():
    if not line.strip():
        continue
    msg = json.loads(line)
    if "error" in msg:
        print("FAIL", msg["error"])
        sys.exit(2)
    if msg.get("id") == 3:
        result = msg.get("result") or {}
        if result.get("success") and (result.get("data") or {}).get("image_png_b64"):
            print("LOCAL_ENHANCE_OK bytes", len(result["data"]["image_png_b64"]))
            ok = 1
        else:
            print("FAIL enhance", result)
            sys.exit(3)
if not ok:
    print("FAIL missing enhance result")
    sys.exit(4)
print("CLOUD_LINUX_FRESH_SMOKE_PASS")
PY
"$WORKDIR/local-out.jsonl"
