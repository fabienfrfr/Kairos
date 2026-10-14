#!/usr/bin/env bash
# CPU torch without NVIDIA wheels (stub CUDA libs); run, then: source "$HOME/torch_env.sh"
set -euo pipefail

DIR="${SANDBOX_TORCH_DIR:-$HOME}"
VENV="$DIR/venv"
STUBS="$DIR/stubs"
PY="$VENV/bin/python"
# the repo pyproject sets extra uv indexes (ROCm) that are unreachable here
UV="uv --no-config"
PKGS="${EXTRA_PKGS:-transformers datasets typer pyyaml einops accelerate tensorboard huggingface-hub pytest}"
CUDA_LIBS='^lib(cu|nccl|nvrtc|nvshmem|nvJit|nvToolsExt|cupti)'
CUDA_SYMS='^(__cu|__nv|cu|nccl|nvrtc|nvJitLink|cudnn|cublas|cusparse|cufft|curand|cusolver|nvtx|nvshmem|cufile|cupti|nvidia)'
NO_DEVICE='cudaGetDeviceCount cuInit cuDeviceGetCount cudaGetDevice cudaGetDeviceProperties cudaGetDeviceProperties_v2'

echo "[1/6] freeing disk and removing leftover partial NVIDIA files"
rm -rf "$HOME/.cache/pip" "$HOME/.cache/uv" /tmp/pip-* 2>/dev/null || true
rm -rf /usr/local/lib/python3*/dist-packages/nvidia 2>/dev/null || true
avail_kb=$(df --output=avail -k "$DIR" | tail -1 | tr -d ' ')
[ "$avail_kb" -gt 3000000 ] || echo "WARNING: under 3 GB free, the install may fail"

echo "[2/6] venv"
[ -x "$PY" ] || $UV venv --system-site-packages -q "$VENV"

echo "[3/6] torch alone (--no-deps: pulling the NVIDIA wheels would fill the disk)"
if [ ! -f "$DIR/.done_torch" ]; then
  $UV pip install --python "$PY" --no-cache --no-deps torch
  touch "$DIR/.done_torch"
fi
TORCH_LIB=$("$PY" -c "import importlib.util as u, pathlib; print(pathlib.Path(u.find_spec('torch').origin).parent / 'lib')")

echo "[4/6] stub CUDA libraries"
mkdir -p "$STUBS"
nm -D --undefined-only "$TORCH_LIB"/*.so 2>/dev/null | awk '{print $2}' | grep -E "$CUDA_SYMS" | sed 's/@.*//' | sort -u > "$STUBS/syms.txt"
NO_DEVICE="$NO_DEVICE" python3 - "$STUBS" <<'PYEOF'
import os
import sys

no_device = set(os.environ["NO_DEVICE"].split())
stubs = sys.argv[1]
syms = open(f"{stubs}/syms.txt").read().split()
with open(f"{stubs}/stub.c", "w") as f:
    f.writelines(f"int {s}(void){{return {100 if s in no_device else 0};}}\n" for s in syms)
PYEOF
for f in "$TORCH_LIB"/*.so; do readelf -d "$f" 2>/dev/null | awk '/NEEDED/ {gsub(/[\[\]]/, "", $5); print $5}'; done \
  | sort -u | grep -E "$CUDA_LIBS" | while read -r lib; do
    node="$lib"
    case "$lib" in libnvshmem*) node="NVSHMEM" ;; esac
    printf '%s {\n global: *;\n};\n' "$node" > "$STUBS/$lib.map"
    gcc -shared -fPIC -Wl,-soname,"$lib" -Wl,--version-script="$STUBS/$lib.map" -o "$STUBS/$lib" "$STUBS/stub.c"
  done

echo "[5/6] other dependencies, NVIDIA/triton filtered out"
if [ ! -f "$DIR/.done_deps" ]; then
  # shellcheck disable=SC2086
  $UV pip install --python "$PY" --dry-run torch $PKGS 2>&1 | grep -E '^ \+ ' | awk '{print $2}' \
    | grep -v -E '^(nvidia-|triton|torch==|cuda-)' > "$DIR/pkgs.txt"
  $UV pip install --python "$PY" --no-cache --no-deps -r "$DIR/pkgs.txt"
  touch "$DIR/.done_deps"
fi

echo "[6/6] verification"
cat > "$DIR/torch_env.sh" <<ENVEOF
export LD_LIBRARY_PATH="$STUBS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=""
export PATH="$VENV/bin:\$PATH"
ENVEOF
# shellcheck disable=SC1091
. "$DIR/torch_env.sh"
"$PY" -c "
import torch
assert not torch.cuda.is_available()
print('torch', torch.__version__, 'CPU ok:', (torch.ones(2, 2) @ torch.ones(2, 2)).sum().item() == 8.0)"
echo "ready: source $DIR/torch_env.sh && python -m pytest -q -m 'not integration'"
