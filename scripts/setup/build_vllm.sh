#!/usr/bin/env bash
# Build the vLLM editable install for the SM75 (2x2080 Ti) deployment.
#
# - The CUDA toolkit comes from the pip `nvidia-cu13` package, not the system
#   one (the system nvcc here is 12.8).
# - Only SM 7.5 is compiled (single-machine deployment).
# - Sources under .deps are pre-populated, so FetchContent runs disconnected.
# - External hosts go through $PROXY; the PyPI mirror goes direct.
#
# The preflight below checks the five repairs this checkout needs but cannot
# create by itself; docs/upstream-branch.md §3 records how each was produced:
#   1. nvidia-cuda-nvcc / nvvm / crt pinned to 13.0.x (13.3 headers do not
#      match the torch cu130 build);
#   2. unversioned .so symlinks next to the versioned ones in the toolkit lib;
#   3. lib64 -> lib;
#   4. stale source paths inside .deps rewritten to this checkout (a moved
#      checkout otherwise makes cmake look for the old directory);
#   5. cutlass cloned into .deps/cutlass-src.
#
# Paths come from .env (MODEL_PATH / VLLM_PYTHON / FLASHQLA_PATH) plus
# VLLM_SRC, which is not part of .env.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -P "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
if [ -f "$REPO_ROOT/.env" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
fi

: "${VLLM_SRC:?set VLLM_SRC to the vLLM checkout to build}"
: "${VLLM_PYTHON:?set VLLM_PYTHON in .env}"
: "${CUDA_TOOLKIT_PKG:=cu13}"
: "${TORCH_CUDA_ARCH_LIST:=7.5}"
: "${MAX_JOBS:=6}"
: "${PROXY:=}"
: "${PIP_INDEX_URL:=https://mirrors.aliyun.com/pypi/simple/}"

[ -d "$VLLM_SRC/.git" ] || [ -f "$VLLM_SRC/pyproject.toml" ] || {
  echo "error: $VLLM_SRC does not look like a vLLM checkout" >&2
  exit 1
}
[ -x "$VLLM_PYTHON" ] || { echo "error: VLLM_PYTHON=$VLLM_PYTHON is not executable" >&2; exit 1; }

VENV_ROOT=$(dirname "$(dirname "$VLLM_PYTHON")")
CUDA_HOME=$(ls -d "$VENV_ROOT"/lib/python*/site-packages/nvidia/"$CUDA_TOOLKIT_PKG" 2>/dev/null | head -1 || true)
[ -n "$CUDA_HOME" ] || {
  echo "error: nvidia-$CUDA_TOOLKIT_PKG not found in $VENV_ROOT (pip install it first)" >&2
  exit 1
}
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "error: no nvcc in $CUDA_HOME" >&2; exit 1; }

rc=0
vers=$("$CUDA_HOME/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -1)
case "$vers" in
  13.0) ;;
  *) echo "[warn] nvcc is $vers, expected 13.0.x (see docs/upstream-branch.md §3)" >&2; rc=1 ;;
esac
for so in libcudart libnvrtc libcublas; do
  if ! ls "$CUDA_HOME"/lib/${so}.so >/dev/null 2>&1; then
    echo "[warn] missing unversioned $CUDA_HOME/lib/${so}.so (ln -s ${so}.so.* ${so}.so)" >&2
    rc=1
  fi
done
[ -e "$CUDA_HOME/lib64" ] || { echo "[warn] missing $CUDA_HOME/lib64 symlink" >&2; rc=1; }
[ -d "$VLLM_SRC/.deps/cutlass-src" ] || {
  echo "[warn] $VLLM_SRC/.deps/cutlass-src is missing (clone cutlass v4.7.1 there)" >&2
  rc=1
}
stale=$(grep -rhoE "/[A-Za-z0-9_./-]*/\.deps/[A-Za-z0-9_.-]+" "$VLLM_SRC/.deps" 2>/dev/null \
        | sort -u | while read -r p; do [ -e "$p" ] || echo "$p"; done | head -3 || true)
if [ -n "$stale" ]; then
  echo "[warn] .deps references paths that do not exist:" >&2
  printf '       %s\n' $stale >&2
  echo "       rewrite the stale checkout path inside .deps (see docs/upstream-branch.md §3)" >&2
  rc=1
fi
[ "$rc" -eq 0 ] || echo "[warn] preflight found issues; building anyway" >&2
if [ "${CHECK_ONLY:-0}" = "1" ]; then
  echo "[check] preflight done (rc=$rc)"
  exit "$rc"
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$(dirname "$VLLM_PYTHON"):$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST MAX_JOBS
export VLLM_USE_PRECOMPILED=0
export CMAKE_ARGS="-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
if [ -n "$PROXY" ]; then
  export http_proxy="$PROXY" https_proxy="$PROXY"
fi
export no_proxy="${no_proxy:-mirrors.aliyun.com,.aliyun.com,127.0.0.1,localhost}"

exec "$VLLM_PYTHON" -m pip install -e "$VLLM_SRC" --no-build-isolation \
  --index-url "$PIP_INDEX_URL"
