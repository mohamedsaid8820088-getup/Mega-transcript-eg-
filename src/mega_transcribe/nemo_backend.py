"""Local NeMo ASR backend. Heavy NeMo dependencies are imported lazily."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def transcribe_nemo(
    audio_path: Path,
    model_path: Path,
    output_dir: Path,
    *,
    language: str | None = None,
) -> list[Path]:
    """Transcribe a local audio file with a local .nemo ASR model."""
    audio_path = audio_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"ملف الصوت غير موجود: {audio_path}")
    if model_path.suffix.lower() != ".nemo" or not model_path.is_file():
        raise FileNotFoundError(f"ملف نموذج NeMo (.nemo) غير موجود: {model_path}")
    if language and language.lower() not in {"ar", "arabic", "egyptian"}:
        raise ValueError("النموذج الظاهر موجّه للعربية المصرية؛ استخدم --language ar أو اترك اللغة غير محددة.")

    try:
        import torch
        from nemo.collections.asr.models import ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "محرك NeMo غير مثبت أو غير متوافق. ثبّت PyTorch وnemo_toolkit[asr] في بيئة متوافقة "
            "مع Python/ARM64؛ راجع README.md. لم يتم تنزيل أي ملفات."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = ASRModel.restore_from(restore_path=str(model_path), map_location=torch.device("cpu"))
        model.eval()
        with torch.inference_mode():
            predictions = model.transcribe(
                paths2audio_files=[str(audio_path)],
                batch_size=1,
                return_hypotheses=False,
            )
    except Exception as exc:
        raise RuntimeError(
            f"فشل تحميل/تشغيل نموذج NeMo محليًا ({type(exc).__name__}): {exc}. "
            "تحقق من توافق نسخة النموذج مع NeMo ووجود مساحة وذاكرة كافية."
        ) from exc

    if not predictions:
        raise RuntimeError("لم يُرجع نموذج NeMo أي نتيجة.")
    first: Any = predictions[0]
    if isinstance(first, str):
        text = first.strip()
    elif hasattr(first, "text"):
        text = str(first.text).strip()
    else:
        text = str(first).strip()

    txt_path = output_dir / f"{audio_path.stem}.txt"
    json_path = output_dir / f"{audio_path.stem}.json"
    txt_path.write_text(text + "\n", encoding="utf-8")
    payload = {
        "audio": str(audio_path),
        "model": str(model_path),
        "language": language or "ar (model default)",
        "text": text,
        "timestamps_available": False,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [txt_path, json_path]
