#!/usr/bin/env bash
set -euo pipefail

# Paths (edit if your layout is different)
LLVM_ROOT="${LLVM_ROOT:-$HOME/mlir-learning/llvm-project}"
CH6_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)/toy-ch6"

if [[ ! -d "$LLVM_ROOT/mlir/examples/toy/Ch6" ]]; then
  echo "Error: Can't find $LLVM_ROOT/mlir/examples/toy/Ch6"
  echo "Set LLVM_ROOT=/path/to/llvm-project and retry."
  exit 1
fi

echo "[1/3] Copying toy-ch6 sources into llvm-project…"
rsync -a --delete "$CH6_SRC/" "$LLVM_ROOT/mlir/examples/toy/Ch6/"

echo "[2/3] Configuring (if needed)…"
mkdir -p "$LLVM_ROOT/build"
cd "$LLVM_ROOT/build"
# Configure if not already configured
if [[ ! -f CMakeCache.txt ]]; then
  cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DCMAKE_BUILD_TYPE=Release
fi

echo "[3/3] Building toyc-ch6…"
ninja toyc-ch6

echo "Done!"
echo "Binary: $LLVM_ROOT/build/bin/toyc-ch6"
