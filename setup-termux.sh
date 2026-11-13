#!/data/data/com.termux/files/usr/bin/bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${TERMUX_VERSION:-}" && "${PREFIX:-}" != /data/data/com.termux/files/usr ]]; then
  echo "شغّل هذا السكريبت داخل تطبيق Termux نفسه، وليس Horus/Debian." >&2
  exit 1
fi
command -v pkg >/dev/null || { echo "مدير pkg الخاص بـTermux غير متاح." >&2; exit 1; }

echo "المساحة المتاحة قبل التثبيت:"
df -h "$PREFIX" || true
cat <<'EOF'
سيتم تثبيت/تحديث أدوات Termux الأساسية فقط: Python وFFmpeg وGit.
لن يتم تشغيل pip أو ترقية pip، ولن تُثبّت حزم NeMo/PyTorch أو ONNX أو llama.cpp الثقيلة تلقائيًا.
EOF
read -r -p "متابعة؟ [y/N] " answer
case "$answer" in y|Y|yes|YES) ;; *) echo "تم الإلغاء دون تثبيت."; exit 0 ;; esac

pkg update -y
pkg install -y python ffmpeg git
mkdir -p "$ROOT/models/AudioModels" "$ROOT/inputs" "$ROOT/outputs" "$ROOT/bin"

cat > "$ROOT/bin/mega-transcribe" <<EOF
#!/data/data/com.termux/files/usr/bin/bash
ROOT='$ROOT'
export MEGA_PROJECT_DIR="\$ROOT"
export PYTHONPATH="\$ROOT/src\${PYTHONPATH:+:\$PYTHONPATH}"
exec python -m mega_transcribe "\$@"
EOF
cat > "$ROOT/bin/mega-transcribe-web" <<EOF
#!/data/data/com.termux/files/usr/bin/bash
ROOT='$ROOT'
export MEGA_PROJECT_DIR="\$ROOT"
export PYTHONPATH="\$ROOT/src\${PYTHONPATH:+:\$PYTHONPATH}"
exec python -m mega_transcribe.web_app "\$@"
EOF
chmod +x "$ROOT/bin/mega-transcribe" "$ROOT/bin/mega-transcribe-web"
ln -sf "$ROOT/bin/mega-transcribe" "$PREFIX/bin/mega-transcribe"
ln -sf "$ROOT/bin/mega-transcribe-web" "$PREFIX/bin/mega-transcribe-web"

cat <<EOF

اكتمل إعداد الأدوات الأساسية دون استخدام pip.
المشروع: $ROOT
الخطوات التالية:
  1) termux-setup-storage (مرة واحدة إذا لم تنفذه)
  2) mega-transcribe doctor --models-dir "\$HOME/storage/shared/Download/AudioModels"
  3) mega-transcribe menu
  4) للواجهة: mega-transcribe-web --host 127.0.0.1 --port 8765

NeMo/PyTorch وllama.cpp وملفات GGUF تحتاج تحققًا منفصلًا من التوافق والمساحة.
EOF
