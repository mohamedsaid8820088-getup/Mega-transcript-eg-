"""CLI with local model discovery and diagnostics."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "AudioModels"
NEMO_NAME = "asr-egyptian-nemo-v2.0.nemo"
QWEN_NAME = "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"


def _default_nemo(models_dir: Path) -> Path:
    env_value = os.environ.get("MEGA_NEMO_MODEL")
    if env_value:
        return Path(env_value).expanduser()
    direct = models_dir / "asr-egyptian-nemo-v2.0" / NEMO_NAME
    if direct.exists():
        return direct
    found = list(models_dir.rglob(NEMO_NAME)) if models_dir.exists() else []
    return found[0] if found else direct


def _default_qwen() -> Path:
    env_value = os.environ.get("MEGA_QWEN_MODEL")
    return Path(env_value).expanduser() if env_value else PROJECT_ROOT / "models" / QWEN_NAME


def _doctor(models_dir: Path) -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.platform})")
    print(f"Architecture: {os.uname().machine if hasattr(os, 'uname') else 'unknown'}")
    print(f"Models directory: {models_dir}")
    nemo = _default_nemo(models_dir)
    qwen = _default_qwen()
    print(f"NeMo ASR: {'FOUND' if nemo.is_file() else 'MISSING'} — {nemo}")
    print(f"Silero VAD ONNX: {'FOUND' if (models_dir / 'vad' / 'silero_vad.onnx').is_file() else 'MISSING'} — {models_dir / 'vad' / 'silero_vad.onnx'}")
    print(f"MossFormer weights: {'FOUND' if (models_dir / 'MossFormer2-SE-6bit' / 'model.safetensors').is_file() else 'MISSING'} — {models_dir / 'MossFormer2-SE-6bit' / 'model.safetensors'}")
    print(f"Qwen GGUF: {'FOUND' if qwen.is_file() else 'MISSING'} — {qwen}")
    for module in ("torch", "nemo", "onnxruntime", "llama_cpp"):
        try:
            __import__(module)
            status = "installed"
        except ImportError:
            status = "not installed (optional)"
        except Exception as exc:
            status = f"import error ({type(exc).__name__})"
        print(f"{module}: {status}")
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        print(f"Address-space limit: {soft if soft > 0 else 'unlimited'} bytes")
    except (ImportError, OSError, ValueError):
        pass
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        print(f"Project filesystem free: {usage.free // (1024 * 1024)} MiB")
    except OSError:
        pass
    print("No model is downloaded or run by doctor.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mega-transcribe", description="Offline transcription and local coding assistant for Horus.")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect paths, optional dependencies, and available disk space.")
    doctor.add_argument("--models-dir", type=Path, default=Path(os.environ.get("MEGA_MODELS_DIR", str(DEFAULT_MODELS_DIR))))

    transcribe = commands.add_parser("transcribe", help="Transcribe audio using a local NeMo .nemo model.")
    transcribe.add_argument("audio", type=Path)
    transcribe.add_argument("--model", type=Path, default=None)
    transcribe.add_argument("--models-dir", type=Path, default=Path(os.environ.get("MEGA_MODELS_DIR", str(DEFAULT_MODELS_DIR))))
    transcribe.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    transcribe.add_argument("--language", default="ar")

    assistant = commands.add_parser("assistant", help="Ask a local Qwen GGUF model for coding help.")
    assistant.add_argument("--model", type=Path, default=_default_qwen())
    assistant.add_argument("--prompt", help="Question, code snippet, or error description; omit for interactive mode.")
    assistant.add_argument("--context-size", type=int, default=2048)
    assistant.add_argument("--threads", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(args.models_dir.expanduser().resolve())
        if args.command == "transcribe":
            model = args.model or _default_nemo(args.models_dir)
            from .nemo_backend import transcribe_nemo
            for path in transcribe_nemo(args.audio, model, args.output_dir, language=args.language):
                print(f"Saved: {path}")
            return 0
        if args.command == "assistant":
            if args.context_size < 512 or args.threads < 1:
                parser.error("--context-size must be >=512 and --threads must be >=1")
            from .assistant import ask
            if args.prompt:
                print(ask(args.model, args.prompt, context_size=args.context_size, threads=args.threads))
                return 0
            print("Local Qwen assistant. Suggestions are not executed. Type /exit to quit.")
            while True:
                try:
                    prompt = input("\nYou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if prompt.lower() in {"/exit", "/quit"}:
                    break
                if prompt:
                    print("\nQwen> " + ask(args.model, prompt, context_size=args.context_size, threads=args.threads))
            return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"خطأ: {exc}", file=sys.stderr)
        return 1
    return 2
