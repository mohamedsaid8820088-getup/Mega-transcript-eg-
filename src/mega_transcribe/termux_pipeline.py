"""Termux-oriented local conversion and NeMo transcription.

Lossy codecs cannot recover discarded information. Originals are never modified.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        value = float(result.stdout.strip())
        return value if value >= 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _text(hypothesis: Any) -> str:
    value = getattr(hypothesis, "text", None)
    return str(value if value is not None else hypothesis).strip()


def _word_cues(hypothesis: Any) -> list[tuple[float, float, str]]:
    """Return only timestamps actually emitted by the model."""
    data = getattr(hypothesis, "timestamp", None)
    if not isinstance(data, dict):
        return []
    entries = data.get("word") or data.get("words")
    if not isinstance(entries, (list, tuple)):
        return []
    cues: list[tuple[float, float, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
            token = str(item.get("word", item.get("token", ""))).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if token and 0 <= start < end:
            cues.append((start, end, token))
    return cues


def _transcribe(model: Any, audio: Path) -> Any:
    try:
        result = model.transcribe(paths2audio_files=[str(audio)], batch_size=1, return_hypotheses=True)
    except TypeError:
        # Older NeMo releases may not accept return_hypotheses.
        result = model.transcribe(paths2audio_files=[str(audio)], batch_size=1)
    if isinstance(result, tuple):
        result = result[0]
    if not result:
        raise RuntimeError("The NeMo model returned no transcription.")
    return result[0]


def transcribe(
    audio_path: Path,
    model_path: Path,
    output_dir: Path,
    *,
    language: str | None = "ar",
    profile: str = "faithful",
    formats: tuple[str, ...] = ("txt", "srt", "json"),
) -> list[Path]:
    """Convert a temporary audio copy to ASR format and run the local .nemo model.

    faithful: codec decode/resample/channel conversion only.
    speech: optional FFmpeg high/low-pass and light denoise; may remove speech detail.
    """
    audio_path = audio_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if model_path.suffix.lower() != ".nemo" or not model_path.is_file():
        raise FileNotFoundError(f"NeMo .nemo model not found: {model_path}")
    if profile not in {"faithful", "speech"}:
        raise ValueError("profile must be 'faithful' or 'speech'")
    allowed = {"txt", "srt", "json"}
    if not formats or set(formats) - allowed:
        raise ValueError("formats must contain one or more of txt, srt, json")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is missing. In Termux run: pkg install ffmpeg")
    if language and language.lower() not in {"ar", "arabic", "egyptian", "en", "english"}:
        raise ValueError("Unsupported language option. Use ar, en, or leave empty for model default.")

    try:
        import torch
        from nemo.collections.asr.models import ASRModel
    except ImportError as exc:
        raise RuntimeError("NeMo/PyTorch are not installed or unsupported on this Termux build. See README Termux compatibility notes.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mega-audio-") as temp_dir:
        pcm_path = Path(temp_dir) / "input.wav"
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio_path), "-vn"]
        if profile == "speech":
            command += ["-af", "highpass=f=70,lowpass=f=7600,afftdn=nf=-25"]
        command += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(pcm_path)]
        try:
            subprocess.run(command, capture_output=True, text=True, timeout=3600, check=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()[-1500:]
            raise RuntimeError(f"FFmpeg could not decode/convert the audio: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Audio conversion exceeded the one-hour safety timeout.") from exc

        try:
            model = ASRModel.restore_from(restore_path=str(model_path), map_location=torch.device("cpu"))
            model.eval()
            with torch.inference_mode():
                hypothesis = _transcribe(model, pcm_path)
        except Exception as exc:
            raise RuntimeError(f"NeMo transcription failed ({type(exc).__name__}): {exc}") from exc

    text = _text(hypothesis)
    cues = _word_cues(hypothesis)
    duration = _duration(audio_path)
    if not cues and text:
        # Honest clip-level fallback; never fabricate word alignment.
        end = duration if duration and duration > 0 else 0.1
        cues = [(0.0, end, text)]
    stem = audio_path.stem
    result: list[Path] = []
    if "txt" in formats:
        path = output_dir / f"{stem}.txt"
        path.write_text(text + "\n", encoding="utf-8")
        result.append(path)
    if "srt" in formats:
        path = output_dir / f"{stem}.srt"
        body = "\n\n".join(f"{i}\n{_timestamp(start)} --> {_timestamp(end)}\n{token}" for i, (start, end, token) in enumerate(cues, 1))
        path.write_text((body + "\n") if body else "", encoding="utf-8")
        result.append(path)
    if "json" in formats:
        path = output_dir / f"{stem}.json"
        payload = {
            "audio": str(audio_path), "model": str(model_path), "language": language or "model default",
            "preprocessing_profile": profile, "duration_seconds": duration, "text": text,
            "segments": [{"start": s, "end": e, "text": t} for s, e, t in cues],
            "timestamp_precision": "word-level model timestamps" if _word_cues(hypothesis) else "whole-clip fallback; not word-aligned",
            "warning": "Lossy codec information cannot be reconstructed. Review transcription and timestamps.",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result.append(path)
    return result
