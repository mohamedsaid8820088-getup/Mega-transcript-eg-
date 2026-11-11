#!/data/data/com.termux/files/usr/bin/bash
set -Eeuo pipefail

# Optional local Qwen runtime. This builds llama.cpp in Termux; allow ample free
# storage, keep the device cool, and do not run other memory-heavy apps.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${LLAMA_CPP_DIR:-$ROOT/.build/llama.cpp}"

if [[ -z "${TERMUX_VERSION:-}" && "${PREFIX:-}" != /data/data/com.termux/files/usr ]]; then
  echo "Run this script from Termux." >&2
  exit 1
fi
command -v pkg >/dev/null || { echo "Termux pkg manager not found." >&2; exit 1; }
command -v cmake >/dev/null || {
  echo "Build tools are missing. Install only if storage permits: pkg install clang cmake make git" >&2
  exit 1
}
command -v git >/dev/null || { echo "Install git first: pkg install git" >&2; exit 1; }

echo "Free space:"; df -h "$PREFIX"
read -r -p "Building llama.cpp may need substantial storage/RAM. Continue? [y/N] " answer
case "$answer" in y|Y|yes|YES) ;; *) echo "Cancelled."; exit 0 ;; esac

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  mkdir -p "$(dirname "$LLAMA_DIR")"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$LLAMA_DIR/build" --target llama-server -j2
printf '\nBuilt server: %s\n' "$LLAMA_DIR/build/bin/llama-server"
printf 'Set MEGA_LLAMA_SERVER to that path and MEGA_QWEN_MODEL to your local .gguf file.\n'
