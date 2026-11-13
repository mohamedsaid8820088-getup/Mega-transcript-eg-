from pathlib import Path

from mega_transcribe.model_inventory import inventory


def test_inventory_distinguishes_detected_weights_from_integrated_backends(tmp_path: Path):
    ctc = tmp_path / "asr_egyptian_ctc" / "model_int8.onnx"
    ctc.parent.mkdir()
    ctc.write_bytes(b"onnx")
    vad = tmp_path / "vad" / "silero_vad.onnx"
    vad.parent.mkdir()
    vad.write_bytes(b"onnx")
    moss = tmp_path / "MossFormer2-SE-6bit" / "model.safetensors"
    moss.parent.mkdir()
    moss.write_bytes(b"weights")
    nemo = tmp_path / "asr-egyptian-nemo-v2.0.nemo"
    nemo.write_bytes(b"nemo")

    result = inventory(tmp_path)
    by_name = {item["name"]: item for item in result["models"]}

    assert by_name["asr_egyptian_ctc/model_int8.onnx"]["integrated"] is True
    assert by_name["vad/silero_vad.onnx"]["integrated"] is False
    assert by_name["MossFormer2-SE-6bit/model.safetensors"]["integrated"] is False
    assert by_name["asr-egyptian-nemo-v2.0.nemo"]["integrated"] is True
    assert result["active_pipeline"][0]["allocation_percent"] == 100
    assert result["active_pipeline"][1]["available"] is True
    assert all(item["allocation_percent"] == 0 for item in result["active_pipeline"][2:])


def test_inventory_is_empty_for_missing_model_directory(tmp_path: Path):
    result = inventory(tmp_path / "missing")
    assert result["models"] == []
    assert result["active_pipeline"][0]["available"] is False
