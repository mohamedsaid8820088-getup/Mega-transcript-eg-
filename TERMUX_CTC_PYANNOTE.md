# Termux: تشغيل CTC مع Pyannote segmentation المحلي

الواجهة تقرأ أوزان التاب من `MEGA_MODELS_DIR`، وتشغل CTC و/أو مكوّن Pyannote segmentation اختياريًا. الإعداد الافتراضي:

- `asr_egyptian_ctc/model_int8.onnx` مع `asr_egyptian_ctc/vocab.json` للتفريغ.
- `Pyannote-comunity1.onnx/segmentation-community-1.onnx` لدرجات segmentation الخام.
- `embedding_model.onnx` مفهرس فقط؛ لا تزال معالجة fbank والتجميع غير موصولة.

```bash
export MEGA_MODELS_DIR="$HOME/storage/shared/Download/AudioModels"
export MEGA_CTC_MODEL="$MEGA_MODELS_DIR/asr_egyptian_ctc/model_int8.onnx"
export MEGA_CTC_VOCAB="$MEGA_MODELS_DIR/asr_egyptian_ctc/vocab.json"
export MEGA_PYANNOTE_SEGMENTATION="$MEGA_MODELS_DIR/Pyannote-comunity1.onnx/segmentation-community-1.onnx"
export MEGA_OUTPUT_DIR="$HOME/mega-transcribe-offline/outputs"
cd "$HOME/mega-transcribe-offline"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m mega_transcribe.ctc_web_app --host 127.0.0.1 --port 8765
```

افتح `http://127.0.0.1:8765`، وحدد CTC وPyannote segmentation ثم ارفع عينة قصيرة أولًا. الناتج CTC ملفات TXT/SRT/JSON، وPyannote ملف JSON للاحتمالات الخام لكل إطار. الملف الأصلي لا يتغير.

**الحدود:** فحص ONNX يبين مدخل `input_values` بشكل `[batch, channels, samples]` ومخرج `logits` بسبع فئات. ملفات الجهاز لا تتضمن إعداد pipeline أو تعريف الفئات؛ لذلك لا تُسمى فئات الخرج بهويات، والتوقيتات تقريبية. هذا ليس diarization كاملًا ولا فصلًا صوتيًا. يتطلب استكمال front-end/ميزات embedding المتوافقة، والتجميع والمعايرة. تحذير ONNX Runtime يعني أن Android غير مدعوم رسميًا؛ اختبر كل شيء محليًا على مقطع قصير.
