#!/data/data/com.termux/files/usr/bin/bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${TERMUX_VERSION:-}" && "${PREFIX:-}" != /data/data/com.termux/files/usr ]]; then echo "Run inside Termux." >&2; exit 1; fi
cd "$ROOT"
if ! command -v mega-transcribe >/dev/null 2>&1; then python -m pip install --no-deps -e "$ROOT"; fi
exec mega-transcribe menu
