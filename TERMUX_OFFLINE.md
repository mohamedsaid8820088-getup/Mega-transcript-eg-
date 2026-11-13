# Termux: تشغيل CTC وNeMo محليًا للمقارنة

تدعم واجهة `mega-transcribe-web` اختيار التفريغ المصري CTC ONNX، وNeMo `.nemo`، أو تشغيل الاثنين بالتتابع على نسخة مؤقتة من الملف. تنتج المحركات ملفات منفصلة باسم يتضمن معرف المهمة واسم المحرك؛ لا تدمج النصوص ولا تستبدل ملف المصدر.

## الإعداد اليدوي

```bash
export MEGA_MODELS_DIR="$HOME/storage/shared/Download/AudioModels"
export MEGA_CTC_MODEL="$MEGA_MODELS_DIR/asr_egyptian_ctc/model_int8.onnx"
export MEGA_CTC_VOCAB="$MEGA_MODELS_DIR/asr_egyptian_ctc/vocab.json"
export MEGA_NEMO_MODEL="$HOME/storage/shared/Download/asr-egyptian-nemo-v2.0.nemo"
export MEGA_OUTPUT_DIR="$HOME/mega-transcribe-offline/outputs"
cd "$HOME/mega-transcribe-offline"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m mega_transcribe.ctc_web_app --host 127.0.0.1 --port 8765
```

افتح `http://127.0.0.1:8765` على الجهاز. اختر المحركين للمقارنة إذا كان NeMo وPyTorch مثبتين ومتوافقين مع نسخة Python/Android الحالية. في شاشة الصحة ستظهر حالة الملف وحالة الاستيراد؛ وجود `.nemo` لا يعني أن runtime يعمل. Python 3.14/Termux قد لا يدعمان حزم NeMo/PyTorch المطلوبة. إذا تعذر استيرادهما، استخدم CTC من الواجهة، أو شغّل NeMo على Linux/بيئة Python مدعومة.

المعالجة الافتراضية أمينة ولا تستخدم مرشحًا. خيار مرشح الكلام اختياري، وقد يفقد تفاصيل خافتة. احتفظ بالأصل للمقارنة. Pyannote الموجود ليس فصلًا صوتيًا، وDeepFilter/MossFormer لا يفصلان حديثًا متداخلًا عن تلاوة.

أثناء أول إعداد، يلزم نسخ المستودع والحزم والنماذج يدويًا أو تنزيلها. بعد وجودها محليًا، لا تتصل الواجهة بالشبكة ولا تستخدم سحابة. تحذير ONNX Runtime حول Android يعني أن المنصة غير مدعومة رسميًا؛ اختبر كل تشغيل على عينة قصيرة أولًا.

## الاختبارات

```bash
pytest
```
