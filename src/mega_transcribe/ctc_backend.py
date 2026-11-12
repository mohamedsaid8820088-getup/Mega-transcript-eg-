"""Local Egyptian Arabic Wav2Vec2 CTC inference using ONNX Runtime."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.020
CHUNK_SECONDS = 20
OVERLAP_SECONDS = 1


def _find_asset(env_name: str, models_dir: Path, filename: str) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"الملف المحدد في {env_name} غير موجود: {path}")
        return path
    matches = list(models_dir.rglob(filename)) if models_dir.is_dir() else []
    if not matches:
        raise FileNotFoundError(f"لم أجد {filename} تحت {models_dir}; اضبط {env_name}.")
    return matches[0].resolve()


def default_assets(models_dir: Path | None = None) -> tuple[Path, Path]:
    root = (models_dir or Path(os.environ.get("MEGA_MODELS_DIR", "models/AudioModels"))).expanduser().resolve()
    model = _find_asset("MEGA_CTC_MODEL", root, "model_int8.onnx")
    vocab_env = os.environ.get("MEGA_CTC_VOCAB")
    vocab = Path(vocab_env).expanduser().resolve() if vocab_env else model.parent / "vocab.json"
    if not vocab.is_file():
        raise FileNotFoundError(f"قاموس النموذج غير موجود: {vocab}; اضبط MEGA_CTC_VOCAB.")
    return model, vocab


def _timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _decode_ids(ids: list[int], tokens: dict[int, str]) -> tuple[str, list[tuple[float, float, str]]]:
    """CTC greedy collapse with frame-derived word intervals."""
    chars: list[tuple[int, str]] = []
    previous: int | None = None
    for frame, token_id in enumerate(ids):
        if token_id != previous and token_id != 0:
            token = tokens.get(token_id, "")
            if token and token not in {"<s>", "</s>", "<unk>"}:
                chars.append((frame, token))
        previous = token_id

    text_parts: list[str] = []
    words: list[tuple[float, float, str]] = []
    word_chars: list[tuple[int, str]] = []

    def finish_word(end_frame: int | None = None) -> None:
        if not word_chars:
            return
        value = "".join(token for _, token in word_chars)
        start = word_chars[0][0] * FRAME_SECONDS
        end = max(start + FRAME_SECONDS, (end_frame if end_frame is not None else word_chars[-1][0] + 1) * FRAME_SECONDS)
        words.append((start, end, value))
        word_chars.clear()

    for frame, token in chars:
        if token == "|":
            finish_word(frame)
            if text_parts and text_parts[-1] != " ":
                text_parts.append(" ")
        else:
            word_chars.append((frame, token))
            text_parts.append(token)
    finish_word()
    return "".join(text_parts).strip(), words


def _read_audio(path: Path) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy غير متاح في Python؛ ثبّت حزمة Termux python-numpy.") from exc
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg غير مثبت. في Termux نفّذ: pkg install ffmpeg")
    with tempfile.TemporaryDirectory(prefix="mega-ctc-") as temp:
        pcm = Path(temp) / "audio.wav"
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(pcm)]
        try:
            subprocess.run(command, capture_output=True, text=True, timeout=3600, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"تعذر تحويل الصوت بواسطة FFmpeg: {(exc.stderr or '')[-1200:]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("تجاوز تحويل الصوت مهلة ساعة.") from exc
        with wave.open(str(pcm), "rb") as source:
            if source.getnchannels() != 1 or source.getframerate() != SAMPLE_RATE or source.getsampwidth() != 2:
                raise RuntimeError("تعذر تجهيز الصوت بصيغة PCM أحادي 16 kHz.")
            frames = source.readframes(source.getnframes())
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        raise ValueError("ملف الصوت فارغ أو لا يحتوي عينات صوتية.")
    audio = (audio - float(audio.mean())) / max(float(audio.std()), 1e-7)
    return audio


def _infer(audio: Any, model_path: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime/NumPy غير متاحين. في Termux: pkg install python-onnxruntime python-numpy") from exc
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(os.environ.get("MEGA_CTC_THREADS", "2")))
    options.inter_op_num_threads = 1
    try:
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise RuntimeError(f"تعذر تحميل نموذج ONNX: {type(exc).__name__}: {exc}") from exc
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "audio" or len(outputs) != 1 or outputs[0].name != "logits":
        raise RuntimeError("توقيع النموذج غير متوقع؛ المطلوب input audio وoutput logits.")
    class_dim = outputs[0].shape[-1]
    if isinstance(class_dim, int) and class_dim != 43:
        raise RuntimeError(f"النموذج يعيد {class_dim} فئة، بينما قاموس هذا النموذج متوقع أن يدعم 43 فئة.")
    chunk = CHUNK_SECONDS * SAMPLE_RATE
    overlap = OVERLAP_SECONDS * SAMPLE_RATE
    if audio.size <= chunk:
        logits = session.run(["logits"], {"audio": audio[None, :].astype(np.float32)})[0][0]
        return logits, {"chunked": False}
    pieces = []
    stride = chunk - overlap
    starts = list(range(0, audio.size, stride))
    for number, start in enumerate(starts):
        end = min(start + chunk, audio.size)
        sample = audio[start:end]
        logits = session.run(["logits"], {"audio": sample[None, :].astype(np.float32)})[0][0]
        left = 0 if number == 0 else int(round((overlap / 2) / (SAMPLE_RATE * FRAME_SECONDS)))
        right = len(logits) if end == audio.size else len(logits) - int(round((overlap / 2) / (SAMPLE_RATE * FRAME_SECONDS)))
        if right > left:
            pieces.append(logits[left:right])
    return np.concatenate(pieces, axis=0), {"chunked": True, "chunk_seconds": CHUNK_SECONDS, "overlap_seconds": OVERLAP_SECONDS}


def transcribe_ctc(audio_path: Path, output_dir: Path, *, model_path: Path | None = None, vocab_path: Path | None = None, formats: tuple[str, ...] = ("txt", "srt", "json")) -> list[Path]:
    audio_path = audio_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"ملف الصوت غير موجود: {audio_path}")
    allowed = {"txt", "srt", "json"}
    if not formats or set(formats) - allowed:
        raise ValueError("الصيغ المتاحة: txt وsrt وjson")
    if model_path is None or vocab_path is None:
        default_model, default_vocab = default_assets()
        model_path = model_path or default_model
        vocab_path = vocab_path or default_vocab
    model_path, vocab_path = model_path.expanduser().resolve(), vocab_path.expanduser().resolve()
    with vocab_path.open(encoding="utf-8") as source:
        raw_vocab = json.load(source)
    tokens = {int(index): token for token, index in raw_vocab.items()}
    if tokens.get(0) not in {"<pad>", "[PAD]"}:
        raise ValueError("القاموس غير متوافق: فئة CTC blank المتوقعة ذات المعرف 0.")
    if not all(index in tokens for index in range(43)):
        raise ValueError("القاموس لا ��حتوي جميع فئات خرج النموذج (0–42).")
    audio = _read_audio(audio_path)
    logits, chunk_info = _infer(audio, model_path)
    ids = logits.argmax(axis=-1).tolist()
    text, words = _decode_ids(ids, tokens)
    try:
        import numpy as np
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        confidence = float(probabilities.max(axis=-1).mean())
        blank_ratio = float((np.asarray(ids) == 0).mean())
    except Exception:
        confidence, blank_ratio = None, None
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[Path] = []
    stem = audio_path.stem
    if "txt" in formats:
        target = output_dir / f"{stem}.txt"
        target.write_text(text + "\n", encoding="utf-8")
        result.append(target)
    if "srt" in formats:
        target = output_dir / f"{stem}.srt"
        body = "\n\n".join(f"{i}\n{_timestamp(start)} --> {_timestamp(end)}\n{word}" for i, (start, end, word) in enumerate(words, 1))
        target.write_text(body + ("\n" if body else ""), encoding="utf-8")
        result.append(target)
    if "json" in formats:
        target = output_dir / f"{stem}.json"
        payload = {"audio": str(audio_path), "model": str(model_path), "vocabulary": str(vocab_path), "language": "Egyptian Arabic", "text": text, "segments": [{"start": start, "end": end, "text": word} for start, end, word in words], "timestamp_precision": "CTC frame-derived word intervals (20 ms frames); review recommended", "mean_max_probability": confidence, "blank_frame_ratio": blank_ratio, "inference": chunk_info, "warning": "Greedy CTC confidence cannot reliably distinguish speech from noise. Review low-quality/noisy clips; enhancement cannot recover information absent from the recording."}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result.append(target)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Egyptian Arabic ONNX CTC transcription")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--formats", nargs="+", choices=("txt", "srt", "json"), default=("txt", "srt", "json"))
    args = parser.parse_args(argv)
    try:
        for path in transcribe_ctc(args.audio, args.output_dir, model_path=args.model, vocab_path=args.vocab, formats=tuple(args.formats)):
            print(f"حُفظ: {path}")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"خطأ: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
