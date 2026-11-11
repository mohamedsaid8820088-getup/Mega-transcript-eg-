from pathlib import Path

from mega_transcribe.cli import _default_nemo


def test_default_nemo_discovers_model_recursively(tmp_path: Path) -> None:
    model = tmp_path / "asr-egyptian-nemo-v2.0" / "asr-egyptian-nemo-v2.0.nemo"
    model.parent.mkdir(parents=True)
    model.touch()
    assert _default_nemo(tmp_path) == model


def test_default_nemo_returns_expected_path_when_missing(tmp_path: Path) -> None:
    assert _default_nemo(tmp_path) == tmp_path / "asr-egyptian-nemo-v2.0" / "asr-egyptian-nemo-v2.0.nemo"
