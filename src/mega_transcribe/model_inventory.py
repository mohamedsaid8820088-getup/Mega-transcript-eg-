"""Inventory local model files and report actual backend status."""
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
            elif "segmentation-community-1" in key:
                role, integrated, note = "Pyannote segmentation", True, "تشغيل ONNX للدرجات الخام متاح؛ ليس diarization كاملًا أو فصلًا صوتيًا"
            elif "embedding_model" in key and "pyannote" in key:
                role, integrated, note = "Pyannote embedding", False, "الوزن موجود؛ استخراج fbank والتجميع غير مدمجين"
            elif path.suffix.lower() == ".nemo":
                role, integrated, note = "تفريغ NeMo", True, "يتطلب NeMo وPyTorch متوافقين مع الجهاز"
            elif "silero_vad" in path.name.lower() or ("/vad/" in f"/{key}" and path.suffix.lower() == ".onnx"):
                role, integrated, note = "اكتشاف النشاط الصوتي VAD", False, "الوزن موجود؛ backend VAD غير موصول"
            elif "pyannote" in key or "diarization" in key:
                role, integrated, note = "تمييز المتحدثين / Pyannote", False, "المكوّن المرصود لا يكفي وحده لتشغيل pipeline"
            elif "mossformer" in key:
                role, integrated, note = "تحسين صوت MossFormer2", False, "وزن MLX غير متوافق مباشرة مع runtime Termux الحالي"
            elif "deepfilter" in key:
                role, integrated, note = "خفض ضوضاء DeepFilter", False, "يحتاج استخراج الميزات وإعادة البناء المطابقة للتصدير"
            else:
                role, integrated, note = "نموذج محلي غير معروف", False, "مفهرس فقط؛ لم يُفعّل دون runtime واختبار توافق"
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            files.append({"name": relative, "path": str(path), "format": path.suffix.lower(), "bytes": size, "role": role, "integrated": integrated, "note": note})
            if len(files) >= 300:
                break
    ctc = any(x["integrated"] and x["role"] == "التفريغ المصري CTC" for x in files)
    pyannote = any(x["integrated"] and x["role"] == "Pyannote segmentation" for x in files)
    nemo = any(x["integrated"] and x["role"] == "تفريغ NeMo" for x in files)
    return {"models_dir": str(root), "models": files, "active_pipeline": [
        {"id": "ctc", "name": "التفريغ المصري CTC", "available": ctc, "allocation_percent": 100 if ctc else 0, "note": "محرك تفريغ مستقل"},
        {"id": "pyannote", "name": "Pyannote segmentation ONNX", "available": pyannote, "allocation_percent": 100 if pyannote else 0, "note": "درجات خام فقط؛ لا يحدد هوية المتحدث ولا يعزل الصوت"},
        {"id": "nemo", "name": "تفريغ NeMo", "available": nemo, "allocation_percent": 100 if nemo else 0, "note": "محرك مستقل اختياري"},
        {"id": "vad", "name": "VAD", "available": False, "allocation_percent": 0, "note": "غير مفعّل"},
        {"id": "separation", "name": "فصل المصادر", "available": False, "allocation_percent": 0, "note": "لا يوجد نموذج فصل صوت موصول"}], "notice": "درجات Pyannote خام بلا أسماء متحدثين؛ يلزم front-end وتهيئة وتجميع ومعايرة لإكمال diarization."}
