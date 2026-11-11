# Mega Transcribe Offline — Professional Termux workflow

This repository includes a compact local web UI and terminal menu for Android/Termux. It does not claim that every model shown in screenshots has been integrated: the NeMo `.nemo` ASR is the only transcription backend currently called. Other files are inventoried until their exact runtime/model configuration is verified.

## Important constraints

- Galaxy Tab A7 2020 is ARM64 with limited memory. NeMo/PyTorch installation and inference may be unsupported or exceed storage/RAM in Termux. The bootstrap deliberately does **not** install those heavy packages.
- Qwen must be an actual local `.gguf` file. llama.cpp is an optional separate build; a GUI cannot make unsupported kernels/installations work.
- Silero VAD ONNX, Egyptian CTC, Pyannote, MossFormer2-SE safetensors, and DeepFilter are not falsely presented as active. Their matching inference runtimes, architecture/configs, sampling requirements, and licenses must be checked before integrating them.
- Decoding AMR/MP3 to PCM cannot restore audio samples removed by lossy compression. The original upload is preserved; only a temporary mono 16 kHz PCM copy is passed to ASR. Cleanup filters can remove speech detail, so faithful conversion is the default.
- SRT uses word timestamps only if the model actually supplies them. Otherwise it produces a single whole-clip cue and the JSON explicitly labels timestamps as not word-aligned. It cannot guarantee every character or perfect Quran/broadcast transcription; human review is essential.

## Install in Termux

Install Termux from F-Droid or the official GitHub releases, open Termux itself, and grant storage access when asked:

```bash
termux-setup-storage
pkg update -y
pkg install -y git python ffmpeg
```

Clone and bootstrap:

```bash
git clone https://github.com/mohamedsaid8820088-getup/Mega-transcript-eg-.git "$HOME/mega-transcribe-offline"
cd "$HOME/mega-transcribe-offline"
bash setup-termux.sh
```

The bootstrap asks before installing base packages. It does not install NeMo, PyTorch, ONNX Runtime, or llama.cpp automatically. Check free space first with `df -h "$HOME"`.

## Models and storage

After `termux-setup-storage`, inspect the Android shared Download directory:

```bash
ls -lah "$HOME/storage/shared/Download/AudioModels"
find "$HOME/storage/shared/Download/AudioModels" -maxdepth 4 -type f -print
```

Point the app at the path that exists on your device. Keep model assets out of Git:

```bash
export MEGA_MODELS_DIR="$HOME/storage/shared/Download/AudioModels"
export MEGA_NEMO_MODEL="$MEGA_MODELS_DIR/asr-egyptian-nemo-v2.0/asr-egyptian-nemo-v2.0.nemo"
export MEGA_QWEN_MODEL="$HOME/storage/shared/Download/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
```

If Android shared storage paths are unavailable, copy the assets to Termux home with the Android file manager or use Termux's storage permission. Do not copy multi-gigabyte assets unless space is available.

## Use the terminal menu

```bash
cd "$HOME/mega-transcribe-offline"
mega-transcribe doctor
mega-transcribe menu
```

The menu provides diagnostics, one-file transcription, local UI startup, and Qwen prompting. Direct commands:

```bash
mega-transcribe transcribe "$HOME/storage/shared/Download/recording.amr" \
  --model "$MEGA_NEMO_MODEL" --language ar --profile faithful \
  --formats txt srt json --output-dir outputs
```

Profiles: `faithful` only decodes/resamples/channels; `speech` additionally applies conservative FFmpeg high/low-pass plus light denoising and may discard speech cues. The app preserves the original and does not invent model weights. SRT is saved as `.srt` (not `.xsrt`).

## Lightweight browser UI

Start the local-only server:

```bash
mega-transcribe-web --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` on the same device. The interface supports choosing audio, language, faithful vs optional speech filters, output TXT/SRT/JSON, job status, downloads, model inventory, and the optional local code assistant. Upload limit is 48 MiB to avoid excessive memory use on tablets; place larger audio in Termux-readable storage and use the CLI path instead. The server rejects non-loopback bind addresses and checks browser POST origins; do not expose it to LAN or the public internet.

## NeMo/PyTorch

`nemo_toolkit[asr]` and PyTorch support varies by Python version, Android/Termux, and ARM64 wheel availability. The device previously reported Python 3.13 and under 1 GiB free disk; do not run a large pip install until you have verified compatibility and freed storage. Use upstream NeMo/PyTorch installation guidance for a supported environment. If packages cannot install, run the ASR model on compatible Linux hardware rather than claiming the tablet can run it.

## Qwen coding assistant

`llama-cpp-python` is optional and may require a native build. For a CLI llama.cpp build attempt, inspect storage first, install Termux build tools, then:

```bash
pkg install clang cmake make git
bash build-llama-termux.sh
```

This builds a local `llama-server`; the current web assistant endpoint uses the `llama_cpp` Python binding, not that server. Install/use the binding only if a compatible Termux build is available. If not, using llama.cpp's own CLI/server requires a separately configured client integration. The assistant never executes shell commands or modifies source files.

## Tests

On a development machine with Python and pytest:

```bash
python -m pip install -e '.[dev]'
python -m compileall -q src
pytest
```

Tests do not prove model/runtime compatibility on a particular Android device. The app reports failures rather than silently substituting models.
