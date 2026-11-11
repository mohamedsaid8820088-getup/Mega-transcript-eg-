#!/data/data/com.termux/files/usr/bin/bash
set -Eeuo pipefail

# Minimal Termux bootstrap. Heavy ASR frameworks are deliberately not installed
# automatically: NeMo/PyTorch are not reliably available for every Termux/Python
# and Android ARM64 combination and can exhaust device storage.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${TERMUX_VERSION:-}" && "${PREFIX:-}" != /data/data/com.termux/files/usr ]]; then
  echo "Run this script from the Termux app, not Debian/Horus." >&2
  exit 1
fi

command -v pkg >/dev/null || { echo "Termux pkg manager not found." >&2; exit 1; }

echo "Checking free storage before installing base packages..."
df -h "$PREFIX" || true
read -r -p "Install/update Python, FFmpeg and Git in Termux? [y/N] " answer
case "$answer" in y|Y|yes|YES) ;; *) echo "Cancelled; no packages were installed."; exit 0 ;; esac

pkg update -y
pkg install -y python ffmpeg git
python -m pip --version >/dev/null
python -m pip install --upgrade pip
python -m pip install --no-deps -e "$ROOT"
mkdir -p "$ROOT/models/AudioModels" "$ROOT/inputs" "$ROOT/outputs"
cat <<'EOF'

Base Termux setup is complete.
Next:
  1. Put/copy the extracted AudioModels folder under this project's models/ directory,
     or set MEGA_MODELS_DIR to a path Termux can read.
  2. Run: mega-transcribe doctor
  3. Run: mega-transcribe-web --host 127.0.0.1 --port 8765
  4. Open http://127.0.0.1:8765 in the Android browser.

NeMo/PyTorch and llama.cpp are optional, heavy components. This script did not
install them; verify compatibility and free storage before attempting either.
EOF
