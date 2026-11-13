from pathlib import Path
from mega_transcribe.pyannote_onnx import default_segmentation_path

def test_pyannote_segmentation_default_path(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MEGA_PYANNOTE_SEGMENTATION", raising=False)
    expected = (tmp_path / "Pyannote-comunity1.onnx" / "segmentation-community-1.onnx").resolve()
    assert default_segmentation_path(tmp_path) == expected

def test_pyannote_segmentation_path_override(tmp_path: Path, monkeypatch):
    model = tmp_path / "custom.onnx"
    monkeypatch.setenv("MEGA_PYANNOTE_SEGMENTATION", str(model))
    assert default_segmentation_path(tmp_path) == model.resolve()
