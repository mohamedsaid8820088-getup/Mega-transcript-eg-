"""Local model inventory; report actual runtimes separately from detected files."""
from __future__ import annotations

from pathlib import Path

MODEL_SUFFIXES = {".onnx", ".safetensors", ".pt", ".pth", ".gguf", ".nemo"}


def inventory(models_dir: Path, *, ctc_runtime: bool = True) -> dict:
    root = models_dir.expanduser().resolve()
    files = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            key = relative.lower()
            if "asr_egyptian_ctc" in key and path.name == "model_int8.onnx":
                role, integrated, note = "التفريغ المصري CTC", ctc_runtime, "موصول بمحرك ONNX CTC" if ctc_runtime else "اعتماد ONNX Runtime غير متاح"
            elif path.suffix.lower() == ".nemo":
                role, integrated, note = "تفريغ NeMo", True, "يدعمه مسار NeMo في CLI والواجهة؛ يتطلب تثبيت NeMo وPyTorch متوافقين"
            elif "silero_vad" in path.name.lower() or ("/vad/" in f"/{key}" and path.suffix.lower() == ".onnx"):
                role, integrated, note = "اكتشاف النشاط الصوتي VAD", False, "الوزن موجود؛ backend VAD غير موصول بالتدفق الحالي"
            elif "pyannote" in key or "diarization" in key or "segmentation-community" in key:
                role, integrated, note = "تمييز المتحدثين / Pyannote", False, "ملفات النموذج موجودة؛ pipeline والمعالجة والتجميع لم تدمج"
            elif "mossformer" in key:
                role, integrated, note = "تحسين صوت MossFormer2", False, "وزن MLX safetensors موجود؛ غير متوافق مباشرة مع runtime Termux الحالي"
            elif "deepfilter" in key:
                role, integrated, note = "خفض ضوضاء DeepFilter", False, "ملف ONNX يحتاج استخراج الميزات وإعادة البناء المتوافقة مع هذا التصدير"
            else:
                role, integrated, note = "نموذج محلي غير معروف", False, "مفهرس فقط؛ لم يُفعّل دون تعريف runtime واختبار توافق"
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            files.append({"name": relative, "path": str(path), "format": path.suffix.lower(), "bytes": size, "role": role, "integrated": integrated, "note": note})
            if len(files) >= 300:
                break

    active_ctc = any(item["integrated"] and item["role"] == "التفريغ المصري CTC" for item in files)
    has_nemo = any(item["integrated"] and item["role"] == "تفريغ NeMo" for item in files)
    return {
        "models_dir": str(root),
        "models": files,
        "active_pipeline": [
            {"id": "ctc", "name": "التفريغ المصري CTC", "available": active_ctc, "allocation_percent": 100 if active_ctc else 0, "note": "محرك تفريغ اختياري مستقل"},
            {"id": "nemo", "name": "تفريغ NeMo", "available": has_nemo, "allocation_percent": 100 if has_nemo else 0, "note": "تشغيله اختياري بالتوازي للمقارنة؛ ليس وزنًا يمزج مع CTC"},
            {"id": "vad", "name": "VAD", "available": False, "allocation_percent": 0, "note": "غير مفعّل؛ لا يُطبّق على الصوت."},
            {"id": "separation", "name": "فصل المصادر", "available": False, "allocation_percent": 0, "note": "لا يوجد نموذج فصل صوت مناسب موصول حاليًا."},
            {"id": "diarization", "name": "تمييز المتحدثين", "available": False, "allocation_percent": 0, "note": "ملفات Pyannote المرصودة ليست pipeline جاهزة."},
        ],
        "notice": "CTC وNeMo نموذجان مستقلان؛ يمكن مقارنة نصيهما ولا يجري دمج تنبؤاتهما تلقائيًا. توفر ملف .nemo لا يضمن توافق NeMo/PyTorch مع إصدار Python في Termux.",
    }
