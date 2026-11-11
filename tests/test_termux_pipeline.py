from pathlib import Path
from mega_transcribe.termux_pipeline import _timestamp, _word_cues

class Hypothesis:
    timestamp = {"word": [{"start": 0.0, "end": 0.4, "word": "سلام"}]}

def test_timestamp_srt_format() -> None:
    assert _timestamp(61.234) == "00:01:01,234"

def test_word_cues_use_model_timestamps() -> None:
    assert _word_cues(Hypothesis()) == [(0.0, 0.4, "سلام")]

def test_no_word_cues_when_model_has_no_alignment() -> None:
    class Plain:
        text = "نص كامل"
    assert _word_cues(Plain()) == []
