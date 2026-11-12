"""Model inventory and truthful capability status for local assets."""
from __future__ import annotations

from pathlib import Path

MODEL_SUFFIXES = {".onnx", ".safetensors", ".pt", ".pth", ".gguf", ".nemo"}


def inventory(models_dir: Path, *, ctc_runtime: bool = True) -> dict:
    """List local model assets and distinguish detected weights from active backends."""
    root = models_dir.expanduser().resolve()
    files = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            key = relative.lower()
            if "asr_egyptian_ctc" in key and path.name == "model_int8.onnx":
                role, integrated, note = "التفريغ المصري CTC", ctc_runtime, "النموذج النشط للتفريغ المحلي" if ctc_runtime else "اعتماد ONNX Runtime غير متاح"
            elif "silero_vad" in path.name.lower() or ("/vad/" in f"/{key}" and path.suffix.lower() == ".onnx"):
                role, integrated, note = "اكتشاف النشاط الصوتي VAD", False, "الوزن موجود؛ backend VAD غير موصول بالتدفق الحالي"
            elif "pyannote" in key or "diarization" in key or "segmentation-community" in key:
                role, integrated, note = "تمييز المتحدثين / Pyannote", False, "ملفات النموذج موجودة؛ التهيئة والمحاذاة ومكونات Pyannote غير مدمجة"
            elif "mossformer" in key:
                role, integrated, note = "تحسين صوت MossFormer2", False, "وزن safetensors موجود؛ يحتاج runtime متوافقًا، وليس فصلًا مضمونًا للتلاوة عن حديث متراكب"
            elif "deepfilter" in key:
                role, integrated, note = "خفض ضوضاء DeepFilter", False, "الوزن موجود؛ backend ومعالجة ما قبل/بعد النموذج غير مدمجين"
            else:
                role, integrated, note = "نموذج محلي غير معروف", False, "مفهرس فقط؛ لم يُفعّل دون تعريف runtime واختبار توافق"
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            files.append({"name": relative, "path": str(path), "format": path.suffix.lower(), "bytes": size, "role": role, "integrated": integrated, "note": note})
            if len(files) >= 300:
                break

    active_asr = any(item["integrated"] and item["role"] == "التفريغ المصري CTC" for item in files)
    return {
        "models_dir": str(root),
        "models": files,
        "active_pipeline": [
            {"id": "ctc", "name": "التفريغ المصري CTC", "available": active_asr, "allocation_percent": 100 if active_asr else 0, "note": "المحرك الوحيد المستخدم لإخراج النص؛ النسبة توصيف للمسار وليست وزنًا مدرّبًا." if active_asr else "لم يُعثر على نموذج CTC متاح."},
            {"id": "vad", "name": "VAD", "available": False, "allocation_percent": 0, "note": "غير مفعّل؛ لا يُطبّق على الصوت."},
            {"id": "separation", "name": "فصل المصادر / تحسين الصوت", "available": False, "allocation_percent": 0, "note": "غير مفعّل؛ لا توجد نسبة مزج أو قناع وهمي."},
            {"id": "diarization", "name": "تمييز المتحدثين", "available": False, "allocation_percent": 0, "note": "غير مفعّل؛ ملفات ONNX وحدها لا تكفي لتشغيل pipeline Pyannote."},
        ],
        "notice": "لا تمثل النسب أوزانًا قابلة للمقارنة بين نماذج مختلفة. تغييرها بلا تدريب/معايرة قد يفسد التفريغ. لا تُشغّل إلا المحركات المدمجة فعليًا.",
    }
