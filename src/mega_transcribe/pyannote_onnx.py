"""Experimental runner for the local Pyannote Community-1 segmentation ONNX.

This produces raw frame/class scores only, not diarization labels or separated audio.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Callable

SAMPLE_RATE = 16_000
Progress = Callable[[int, str], None]


def default_segmentation_path(models_dir: Path) -> Path:
    configured = os.environ.get("MEGA_PYANNOTE_SEGMENTATION")
    if configured:
        return Path(configured).expanduser().resolve()
    return (models_dir / "Pyannote-comunity1.onnx" / "segmentation-community-1.onnx").expanduser().resolve()


def _read_audio(path: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("يلزم NumPy لتشغيل Pyannote ONNX.") from exc
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg غير مثبت. في Termux نفّذ: pkg install ffmpeg")
    with tempfile.TemporaryDirectory(prefix="mega-pyannote-") as temp:
        pcm_path = Path(temp) / "mono-16k.wav"
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(pcm_path)]
        try:
            subprocess.run(command, capture_output=True, text=True, timeout=3600, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"تعذر تحويل الصوت: {(exc.stderr or '')[-1000:]}") from exc
        with wave.open(str(pcm_path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getframerate() != SAMPLE_RATE or wav.getsampwidth() != 2:
                raise RuntimeError("تعذر تجهيز PCM أحادي 16 kHz.")
            frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if not audio.size:
        raise ValueError("ملف الصوت فارغ.")
    return audio


def analyze_segmentation(audio_path: Path, model_path: Path, output_path: Path, *, threads: int = 2, progress: Progress | None = None) -> Path:
    """Run ONNX and store raw class probabilities; class mapping is not inferred."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime وNumPy مطلوبان لتشغيل Pyannote segmentation.") from exc
    audio_path, model_path, output_path = audio_path.expanduser().resolve(), model_path.expanduser().resolve(), output_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"الملف الصوتي غير موجود: {audio_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"ملف Pyannote segmentation غير موجود: {model_path}")
    if not 1 <= threads <= 4:
        raise ValueError("عدد خيوط CPU يجب أن يكون بين 1 و4.")
    window_seconds = int(os.environ.get("MEGA_PYANNOTE_WINDOW_SECONDS", "10"))
    if not 2 <= window_seconds <= 30:
        raise ValueError("MEGA_PYANNOTE_WINDOW_SECONDS يجب أن يكون بين 2 و30.")
    if progress:
        progress(10, "تحويل الصوت مؤقتًا إلى PCM أحادي 16 kHz")
    audio = _read_audio(audio_path)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    if progress:
        progress(25, "تحميل segmentation-community-1 ONNX")
    try:
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise RuntimeError(f"تعذر تحميل Pyannote ONNX: {type(exc).__name__}: {exc}") from exc
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "input_values" or len(outputs) != 1 or outputs[0].name != "logits":
        raise RuntimeError("توقيع Pyannote غير مطابق؛ المتوقع input_values ثم logits.")
    out_shape = outputs[0].shape
    if isinstance(out_shape[-1], int) and out_shape[-1] != 7:
        raise RuntimeError(f"خرج segmentation يحوي {out_shape[-1]} فئات، والمتوقع 7.")
    declared = inputs[0].shape[-1]
    window_samples = declared if isinstance(declared, int) else window_seconds * SAMPLE_RATE
    if window_samples <= 0:
        raise RuntimeError("طول نافذة الإدخال في ONNX غير صالح.")
    starts = list(range(0, audio.size, window_samples))
    frames_out = []
    for number, start in enumerate(starts):
        segment = audio[start:start + window_samples]
        valid_samples = int(segment.size)
        if valid_samples < window_samples:
            segment = np.pad(segment, (0, window_samples - valid_samples))
        try:
            logits = session.run(["logits"], {"input_values": segment[None, None, :].astype(np.float32)})[0]
        except Exception as exc:
            raise RuntimeError(f"فشل استدلال النافذة {number + 1}: {type(exc).__name__}: {exc}") from exc
        scores = np.asarray(logits)
        if scores.ndim == 3:
            scores = scores[0]
        if scores.ndim != 2 or scores.shape[-1] != 7:
            raise RuntimeError(f"شكل logits غير متوقع: {tuple(np.asarray(logits).shape)}")
        scores = scores.astype(np.float64)
        shifted = scores - scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= np.maximum(probabilities.sum(axis=-1, keepdims=True), 1e-12)
        kept = max(1, round(scores.shape[0] * valid_samples / window_samples))
        frame_seconds = (valid_samples / SAMPLE_RATE) / scores.shape[0]
        for frame_index, row in enumerate(probabilities[:kept]):
            begin = start / SAMPLE_RATE + frame_index * frame_seconds
            frames_out.append({"start": round(begin, 4), "end": round(min(begin + frame_seconds, (start + valid_samples) / SAMPLE_RATE), 4), "class_probabilities": [round(float(x), 6) for x in row]})
        if progress:
            progress(30 + int(60 * (number + 1) / len(starts)), f"استدلال نافذة {number + 1} من {len(starts)}")
    payload = {
        "audio": str(audio_path), "model": str(model_path),
        "component": "pyannote speaker-diarization-community-1 segmentation ONNX",
        "sample_rate": SAMPLE_RATE, "input_window_seconds": window_samples / SAMPLE_RATE,
        "output_classes": 7,
        "class_semantics": "غير محددة؛ لم يتوفر ملف إعداد/خريطة فئات لهذا التصدير.",
        "timestamp_precision": "توقيت إطارات تقريبي مشتق من خرج ONNX، وليس محاذاة معايرة.",
        "result_type": "raw_segmentation_activity_not_diarization", "frames": frames_out,
        "warning": "يشغّل هذا مكوّن segmentation فقط ويعرض درجات الفئات الخام. لا ينتج هويات متحدثين ولا مسارات صوتية منفصلة. يتطلب diarization الكامل إعداد pipeline واستخراج ميزات embedding والتجميع والمعايرة.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    if progress:
        progress(98, "حفظ درجات segmentation الخام")
    return output_path
