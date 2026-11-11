"""Conservative local audio conversion and NeMo transcription.

Lossy codecs cannot have discarded source information reconstructed. We preserve
original uploads and only make a temporary PCM copy for ASR input.
"""

from __future__ import annotations

import json
import re
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


def _probe_duration(path: Path) -> float | None:
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


def _hypothesis_text(hypothesis: Any) -> str:
    text = getattr(hypothesis, "text", None)
    if text is not None:
        return str(text).strip()
    if isinstance(hypothesis, str):
        return hypothesis.strip()
    return str(hypothesis).strip()


def _word_cues(hypothesis: Any) -> list[tuple[float, float, str]]:
    """Use model-supplied word timestamps only; do not fabricate alignment."""
    timestamp_data = getattr(hypothesis, "timestamp", None)
    if not isinstance(timestamp_data, dict):
        return []
    entries = timestamp_data.get("word") or timestamp_data.get("words")
    if not isinstance(entries, (list, tuple)):
        return []
    cues: list[tuple[float, float, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        start, end = entry.get("start"), entry.get("end")
        token = entry.get("word", entry.get("token