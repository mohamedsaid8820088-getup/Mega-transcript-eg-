# Mega Transcribe Offline

نظام طرفية محلي لمعالجة الصوت ومراجعة الشيفرات في Horus/Debian. يهدف إلى العمل بالنماذج المحلية فقط، ولا يرفع الملفات إلى خدمة سحابية أو ينزّل النماذج تلقائيًا.

## النماذج المستهدفة بحسب الصور

- `asr-egyptian-nemo-v2.0.nemo`: نموذج ASR بصيغة NeMo، وهو مسار التفريغ الأساسي في هذا المشروع.
- `vad/silero_vad.onnx`: ملف VAD. يُكتشف ويُفحص؛ لا يُشغّل في خط التفريغ تلقائيًا لأن تهيئة Silero ONNX (حالة النموذج/حجم الإطار/معدل العينة) يجب أن تطابق نسخة الملف تحديدًا.
- `MossFormer2-SE-6bit/model.safetensors`: أوزان تنقية صوت. لا تُطبّق تلقائيًا؛ يلزم كود البنية وإعداداتها المطابقان للأوزان، ولا تكفي `safetensors` وحدها لاستنتاج ذلك بأمان.
- `asr_egyptian_ctc`, `pyannote`, `deepfilter`: تُعرض في تقرير التشخيص؛ لا يُفترض تنسيقها أو واجهة تشغيلها من اسم المجلد فقط.
- `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`: مساعد محلي اختياري لشرح الشيفرة واقتراح التصحيح. منفصل عن ASR ولا يشغّل الأوامر ولا يعدّل الملفات.

## متطلبات مهمة قبل التثبيت

صور الجهاز تُظهر Android ARM64، وHorus يشغّل Debian، مع Python 3.13.5، ونحو 505 MB ذاكرة متاحة وقرابة 973 MB مساحة حرة وفق المخرجات السابقة. حزمة NeMo/PyTorch كبيرة وتوافقها مع Python 3.13 وARM64 غير مضمون؛ قد لا تكفي المساحة أو الذاكرة لتثبيتها أو تشغيل نموذج 438 MB. لا يمكن ضمان التشغيل دون اختبار على الجهاز. استخدم بيئة Python 3.10–3.12 متوافقة إن كانت متاحة، وتحقق من المساحة قبل تثبيت الاعتماديات. لا تنفذ تثبيت NeMo على البيئة الحالية قبل توفير مساحة كافية.

## إعداد المشروع في Horus

```bash
cd /workspace/projects
mkdir -p mega-transcribe-offline
cd mega-transcribe-offline
```

انسخ/استنسخ ملفات هذا المستودع إلى المجلد، ثم أنشئ بيئة المشروع. لا تخلطها مع `venv` العام الموجود في `/workspace/projects`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

المسار المقترح للن��اذج هو `models/AudioModels/` داخل مجلد المشروع. إذا كان Horus لا يرى مجلد Android Download، انسخ المجلد من مدير الملفات أو عبر ميزة مشاركة/استيراد الملفات في Horus؛ Shizuku وحده لا يضمن ربط تخزين Android ببيئة Debian.

```text
models/AudioModels/
├── asr-egyptian-nemo-v2.0/asr-egyptian-nemo-v2.0.nemo
├── vad/silero_vad.onnx
├── MossFormer2-SE-6bit/model.safetensors
└── ...
models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf   # إن كان متوفرًا
```

أو عيّن المسارات دون نقل الملفات:

```bash
export MEGA_MODELS_DIR="/المسار/الذي-يستطيع-Horus-قراءته/AudioModels"
export MEGA_NEMO_MODEL="$MEGA_MODELS_DIR/asr-egyptian-nemo-v2.0/asr-egyptian-nemo-v2.0.nemo"
export MEGA_QWEN_MODEL="/مسار/ملف/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
```

## التشخيص

```bash
mega-transcribe doctor
mega-transcribe doctor --models-dir models/AudioModels
```

يعرض التشخيص المسارات، الاعتماديات الاختيارية، إصدار Python، الذاكرة والمساحة إن أمكن. غياب نموذج عن التشخيص يعني أن المسار غير قابل للقرا��ة من Debian؛ لا يحاول البرنامج البحث في Android أو استخراج ZIP تلقائيًا.

## تثبيت محرك NeMo (اختياري وكبير)

التفريغ يتطلب `nemo_toolkit[asr]` وPyTorch، وهما غير ضمن تثبيت الأساس. راجع توافق NeMo/PyTorch مع إصدار Python ومعمارية ARM64 في بيئتك أولًا. لا يوجد أمر تثبيت واحد مضمون لكل صور Debian على Android. بعد تهيئة بيئة متوافقة وتثبيت NeMo:

```bash
mega-transcribe transcribe recording.wav --model models/AudioModels/asr-egyptian-nemo-v2.0/asr-egyptian-nemo-v2.0.nemo --language ar --output-dir outputs
```

يقوم البرنامج بتحميل ملف `.nemo` محليًا عبر `restore_from` ويحاول إبقاء inference على CPU. لا يطلب ملفات من الإنترنت. إن احتاج النموذج tokenizer/ملفات خارجية أو تعذر تحميله بسبب توافق Torch/NeMo، تظهر رسالة الخطأ دون تنزيل تلقائي.

## مساعد Qwen (اختياري)

ثبّت llama.cpp Python binding بعد التأكد من المساحة وأدوات البناء المتاحة؛ بناء `llama-cpp-python` على ARM64 قد يكون ثقيلًا. إن كان مثبتًا:

```bash
mega-transcribe assistant --model /مسار/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
mega-transcribe assistant --prompt "اشرح الخطأ واقترح تصحيحًا" --model /مسار/qwen.gguf
```

المساعد لا ينفذ شيفرات أو أوامر، ولا يكتب تغييرات على المشروع. راجع اقتراحاته وشغّل الاختبارات بنفسك. ملف Qwen لم يظهر في الصور، لذلك يجب توفير مساره إن كان موجودًا.

## صيغ النتائج

ينشئ التفريغ ملفي UTF-8: نصًا `.txt` وبيانات `.json` في مجلد النتائج. لا ننشئ SRT زائفًا: واجهة NeMo العامة قد تعيد النص فقط دون طوابع زمنية، ولا تُستنتج الطوابع الزمنية من النص.

```bash
mega-transcribe transcribe recording.wav --model /path/to/model.nemo --output-dir outputs --language ar
```

## حدود التنفيذ الحالية

- نموذج NeMo هو backend ASR الوحيد المفعّل؛ ملفات ONNX وCTC وPyannote وMossFormer تظهر في التشخيص، لكن لا يتم الادعاء بأنها مدمجة قبل معرفة واجهة/إعدادات نسختها.
- لا يوجد تسجيل صوتي حي أو واجهة رسومية أو تنزيل نماذج.
- لا يمكن التحقق عن بعد من ملفات الهاتف. نتائج التشغيل تعتمد على إتاحة الملفات والموارد وتوافق الحزم داخل Horus.
