#!/bin/bash
# Build tandem.cpp for AMD GPUs (HIP/ROCm):
#   1. clone llama.cpp and check out the commit tandem.cpp is based on,
#   2. download tandem.patch from ewhacc/tandem.cpp at a pinned commit and check its SHA-256,
#   3. apply tandem.patch, then tandem-amd.patch,
#   4. build llama-server with HIP.
# usage: bash build.sh [dir] [extra cmake args]
#   AMDGPU_TARGETS=gfx1201 (default; set it to your GPU, e.g. gfx1100)
#   ROCM_PATH=/opt/rocm    (default)
#   NO_BUILD=1             only prepare the sources, do not run cmake
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=${1:-tandem-amd}; shift || true
LLAMA_COMMIT=$(cat "$HERE/LLAMA_CPP_COMMIT")
TANDEM_COMMIT=$(cat "$HERE/TANDEM_COMMIT")
TANDEM_SHA256=8a2e5898d4b750127f9fc474a28b7781615724ea1dbac4ac1c35633df1c1fb5d
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
AMDGPU_TARGETS=${AMDGPU_TARGETS:-gfx1201}

[ -d "$DIR" ] || git clone https://github.com/ggml-org/llama.cpp "$DIR"
cd "$DIR"

if git apply --check --reverse "$HERE/tandem-amd.patch" 2>/dev/null; then
  echo "tandem.patch and tandem-amd.patch are already applied"
else
  git checkout -q "$LLAMA_COMMIT"
  curl -fsSL -o tandem.patch "https://raw.githubusercontent.com/ewhacc/tandem.cpp/$TANDEM_COMMIT/tandem.patch"
  echo "$TANDEM_SHA256  tandem.patch" | sha256sum -c -
  if ! git apply --check tandem.patch 2>/dev/null; then
    echo "tandem.patch does not apply to $(git rev-parse --short HEAD) (local changes?)"; exit 1
  fi
  git apply tandem.patch
  git apply "$HERE/tandem-amd.patch"
  rm tandem.patch
  echo "applied tandem.patch and tandem-amd.patch"
fi

[ -n "$NO_BUILD" ] && { echo "sources ready in $(pwd)"; exit 0; }

cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="$AMDGPU_TARGETS" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM_PATH/bin/amdclang" -DCMAKE_CXX_COMPILER="$ROCM_PATH/bin/amdclang++" "$@"
cmake --build build -j --target llama-server
echo "built: $(pwd)/build/bin/llama-server"
