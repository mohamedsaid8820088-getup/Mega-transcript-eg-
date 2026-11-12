# تشغيل التفريغ CTC على Termux

هذه الإضافة توصل نموذج `asr_egyptian_ctc` بواجهة ويب محلية مستقلة عن NeMo وPyTorch.

## المتطلبات

نموذج ONNX وقاموسه موجودان لد��ك تحت:

```bash
$HOME/storage/shared/Download/AudioModels/asr_egyptian_ctc/
```

في Termux ثبّت الاعتماديات المتاحة من مستودعات Termux:

```bash
pkg install python-numpy python-onnxruntime ffmpeg
```

تحقق من مجلد النماذج، ثم شغّل الخادم:

```bash
cd "$HOME/mega-transcribe-offline"
export MEGA_MODELS_DIR="$HOME/storage/shared/Download/AudioModels"
export MEGA_CTC_MODEL="$MEGA_MODELS_DIR/asr_egyptian_ctc/model_int8.onnx"
export MEGA_CTC_VOCAB="$MEGA_MODELS_DIR/asr_egyptian_ctc/vocab.json"
python -m pip install -e .
mega-transcribe-web --host 127.0.0.1 --port 8765
```

افتح `http://127.0.0.1:8765` على الجهاز نفسه، وافحص ظهور «جاهز — ONNX CTC»، ثم ارفع أول ملف يحوي كلامًا واضحًا. نزّل TXT أو SRT أو JSON من الروابط التي تظهر عند اكتمال المهمة. لإيقاف الخادم اضغط Ctrl+C في Termux.

## تشغيل من الطرفية دون المتصفح

```bash
mega-transcribe-ctc "$HOME/storage/shared/Download/recording.wav" --output-dir outputs
```

ويمكن تمرير النموذج والقاموس صراحةً بخياري `--model` و`--vocab`. يدعم SRT أزمنة كلمات مشتقة من إطارات CTC (كل إطار نحو 20 ms)، وليست محاذاة قسرية مستقلة؛ راجعها يدويًا.

## ملاحظات الدقة

- النموذج يشغّل greedy CTC محليًا على CPU، مع تطبيع الصوت إلى متوسط صفر وانحراف معياري واحد، وتحويل الإدخال إلى mono/16 kHz.
- تُعالج التسجيلات الطويلة بمقاطع متداخلة لتخفيف أثر حدود المقاطع.
- النموذج قد يخرج نصًا من الخشخشة أو الموسيقى؛ لا توجد في هذا التكامل كاشف كلام موثوق، لذلك لا تُفسر الثقة الاحتمالية على أنها إثبات لوجود كلام. راجع الضوضاء والهمس والتداخل بالسماع.
- ملف ONNX يعيد 43 فئة؛ يستخدم فك الرموز معرفات القاموس 0–42 فقط، ولا يستطيع إخراج الرموز 43–50 الموجودة بالقاموس. لا يضيف حركات التشكيل.
- لا يمكن لأي تحسين استعادة معلومات كلام لم تُسجل أصلًا أو غطتها الأصوات بالكامل.
- يبقى تطبيق الواجهة القديم في `web_app.py` كما هو؛ نقطة تشغيل `mega-transcribe-web` الجديدة ت��رض واجهة CTC صريحة لتجنب تشغيل NeMo خطأً.
