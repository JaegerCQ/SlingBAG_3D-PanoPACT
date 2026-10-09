#!/usr/bin/env bash
set -euo pipefail
slingbag_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
slingbag_python="${SLINGBAG_PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$slingbag_root/.build"
export CUDA_CACHE_PATH="$slingbag_root/.cache/cuda"
export MPLCONFIGDIR="$slingbag_root/.cache/matplotlib"
export TRITON_CACHE_DIR="$slingbag_root/.cache/triton"
export MAX_JOBS="${MAX_JOBS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
if [[ -n "${SLINGBAG_CUDA_ARCH:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="$SLINGBAG_CUDA_ARCH"
fi
cd -- "$slingbag_root"
if [[ "$#" -eq 0 ]]; then
    set -- reconstruct.py
fi
exec "$slingbag_python" "$@"
