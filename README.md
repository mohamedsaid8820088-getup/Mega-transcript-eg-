# Mega Transcribe Offline — Termux

واجهة محلية خفيفة لتشغيل تفريغ NeMo، وتحويل الصوت بـFFmpeg، ومساعد Qwen GGUF اختياري. ابدأ من Termux على Android ARM64. لا يمكن ضمان الدقة الحرفية أو استرجاع معلومات فُقدت في ترميز AMR/MP3؛ المعالجة لا تولّد الترددات/الحروف المفقودة ولا تغيّر الملف الأصلي.

## الواقع التقني للنماذج الظاهرة

- `asr-egyptian-nemo-v2.0.nemo`: Backend التفريغ المستهدف. يحتاج NeMo Toolkit وPyTorch، لكن تثبيتهما ليس مضمونًا على Termux/Python/ARM64، وغالبًا يتطلب مساحة كبيرة. اختبر التوافق قبل المحاولة.
- `vad/silero_vad.onnx`، وملفات `asr_egyptian_ctc` وPyannote، و`MossFormer2-SE-6bit/model.safetensors` و`deepfilter`: تُفهرس في الواجهة فقط. لم تُفعّل كـVAD/فصل/تنقية تلقائية لأن الصور لا تحدد إصدارات runtime أو إعدادات النموذج والبنية البرمجية المطابقة. تحميل الأوزان وحده ليس تنفيذًا صالحًا، ولا يضبط هذا الإصدار نسبها أو أوزانها.
- Qwen: ملف GGUF ليس واضحًا في الصور. المساعد يعمل فقط عند وجود ملف `.gguf` فعلي وبناء llama.cpp المتوافق.
- SRT: تُستخدم طوابع word-level إن أعادها نموذج NeMo؛ وإلا ينشأ cue واحد بطول التسجيل ويُذكر صراحةً في JSON أن التوقيت تقريبي. لا تُختلق محاذاة كلمة-بكلمة.

## تثبيت الأساس في Termux

ثبّت Termux من مصدر رسمي موثوق، وافتح التطبيق نفسه لا Debian/Horus. امنح صلاحية الوصول للملفات من Android، ثم:

```bash
termux-setup-storage
pkg update -y
pkg install -y git python ffmpeg
```

انسخ المشروع إلى مساحة Termux (مثل `$HOME/mega-transcribe-offline`) أو استنسخه:

```bash
git clone https://github.com/mohamedsaid8820088-getup/Mega-transcript-eg-.git "$HOME/mega-transcribe-offline"
cd "$HOME/mega-transcribe-offline"
bash setup-termux.sh
```

سكريبت الإعداد يثبت Python وFFmpeg وGit فقط؛ لا يثبت NeMo/PyTorch أو Qwen binding تلقائيًا حتى لا يملأ التخزين أو يوحي بتوافق غير مؤكد. اعرف المساحة أولًا بـ`df -h "$HOME"`.

## إتاحة النماذج

بعد `termux-setup-storage` قد يظهر Download عبر:

```text
$HOME/storage/shared/Download/AudioModels
```

تحقق:

```bash
ls -lah "$HOME/storage/shared/Download/AudioModels"
find "$HOME/storage/shared/Download/AudioModels" -maxdepth 4 -type f -print
```

إذا لم ينجح المسار، انسخ المجلد من مدير الملفات إلى مساحة Termux الخاصة. يوصى بنسخ نموذج ASR الضروري فقط توفيرًا للمساحة، ولا تنسخ/ترفع ملفات النماذج إلى GitHub.

اضبط المسارات (عدّلها إلى المسارات الموجودة فعلًا):

```bash
export MEGA_MODELS_DIR="$HOME/storage/shared/Download/AudioModels"
export MEGA_NEMO_MODEL="$MEGA_MODELS_DIR/asr-egyptian-nemo-v2.0/asr-egyptian-nemo-v2.0.nemo"
export MEGA_QWEN_MODEL="$HOME/storage/shared/Download/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
```

يمكن حفظ هذه الأسطر في `~/.bashrc` بعد التأكد من المسارات. لا تضع نماذج أو مسارات غير موجودة في إعداد ثابت.

## التحقق وتشغيل الواجهة

```bash
mega-transcribe doctor
mega-transcribe-web --host 127.0.0.1 --port 8765
```

افتح في متصفح Android: `http://127.0.0.1:8765`. اربط الخادم بـ`127.0.0.1` فقط؛ لا تستخدم `0.0.0.0` على شبكة عامة. ارفع ملف الصوت من الواجهة، مثل AMR/3GP/MP3/WAV. الحد الأقصى للرفع 200 MiB. لإيقاف الخادم: Ctrl+C.

التحويل المؤقت للأصوات إلى mono/16 kHz/PCM S16LE يهدف لقبول مدخلات ASR. يحتفظ التطبيق بالملف الأصلي، ولا يطبّق تطبيعًا أو إزالة ضوضاء افتراضية، لأن ذلك قد يغيّر التسجيل. فك ترميز صيغة مضغوطة لا يعيد عينات/ترددات حذفها الترميز.

### NeMo ASR

بعد التأكد من وجود بيئة متوافقة ومساحة كافية، يجب تثبيت PyTorch و`nemo_toolkit[asr]` وفق تعليمات وإصدارات تدعم جهازك. لا يوجد أمر pip عام مضمون لـAndroid/Termux ARM64/Python الحالي. عند التثبيت:

```bash
mega-transcribe transcribe "$HOME/storage/shared/Recording.amr" \
  --model "$MEGA_NEMO_MODEL" --language ar --output-dir outputs
```

إن فشل تثبيت Torch/NeMo، فإن بديلًا واقعيًا هو تشغيل backend على Linux/PC متوافق ونقل الصوت إليه؛ لا تدّع أن نموذج `.nemo` يعمل لمجرد وجود الملف.

### Qwen code assistant

الخيار الأصلي Python `llama-cpp-python` يحتاج بناء native وقد يفشل/يستهلك مساحة كبيرة. خيار Termux البديل هو بناء llama.cpp اختياريًا:

```bash
pkg install clang cmake make git
bash build-llama-termux.sh
```

بعدها يتوفر `llama-server`؛ هذه النسخة تكتشف/تعرض ملف GGUF، لكن endpoint الحوار المباشر حاليًا يستخدم `llama-cpp-python`، لذا ثبّت binding المتوافق فقط إذا كان متاحًا على جهازك. لا تُنشئ محاولة بناء كبيرة مع أقل من مساحة حرة مناسبة. مساعد Qwen لا ينفذ الأوامر ولا يكت�� تغييرات.

## المخرجات والقيود

لكل صوت `name.txt`, `name.srt`, و`name.json`. SRT ليس مضمونًا حرفيًا أو متزامنًا على مستوى كل كلمة إذا لم يخرج النموذج بيانات توقيت؛ يوضح JSON مصدر التوقيت. الدقة تتأثر بجودة الميكروفون، الضوضاء، الضغط، اللهجة، التداخل وسعة الجهاز. يجب مراجعة التفريغ يدويًا، لا سيما إذا كان للاستخدام الرسمي أو ديني/إذاعي.

الواجهة تعرض النماذج التي وجدتها، لكنها لا تشغل VAD/pyannote/MossFormer/deepfilter/CTC تلقائيًا دون backend مُطابق ومختبر. لا تضبط sliders وهمية أو نسب فصل لا تتحكم في نموذج فعلي.
