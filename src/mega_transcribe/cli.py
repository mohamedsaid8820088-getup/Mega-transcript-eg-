"""CLI with diagnostics and an interactive Termux menu."""
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
    env = os.environ.get("MEGA_NEMO_MODEL")
    if env:
        return Path(env).expanduser()
    direct = models_dir / "asr-egyptian-nemo-v2.0" / NEMO_NAME
    if direct.is_file():
        return direct
    found = list(models_dir.rglob(NEMO_NAME)) if models_dir.exists() else []
    return found[0] if found else direct

def _default_qwen() -> Path:
    env = os.environ.get("MEGA_QWEN_MODEL")
    if env:
        return Path(env).expanduser()
    candidates = [PROJECT_ROOT / "models" / QWEN_NAME]
    models = Path(os.environ.get("MEGA_MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser()
    if models.exists():
        candidates.extend(models.rglob("*.gguf"))
    return next((p for p in candidates if p.is_file() and ("qwen" in p.name.lower() or p == candidates[0])), candidates[0])

def _doctor(models_dir: Path) -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.platform})")
    print(f"Architecture: {os.uname().machine if hasattr(os, 'uname') else 'unknown'}")
    print(f"Models directory: {models_dir}")
    nemo, qwen = _default_nemo(models_dir), _default_qwen()
    print(f"NeMo ASR: {'FOUND' if nemo.is_file() else 'MISSING'} — {nemo}")
    for label, path in (("Silero VAD", models_dir / "vad" / "silero_vad.onnx"), ("MossFormer weights", models_dir / "MossFormer2-SE-6bit" / "model.safetensors"), ("Qwen GGUF", qwen)):
        print(f"{label}: {'FOUND' if path.is_file() else 'MISSING'} — {path}")
    for module in ("torch", "nemo", "onnxruntime", "llama_cpp"):
        try:
            __import__(module); status = "installed"
        except ImportError: status = "not installed (optional)"
        except Exception as exc: status = f"import error ({type(exc).__name__})"
        print(f"{module}: {status}")
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        print(f"Project filesystem free: {usage.free // (1024*1024)} MiB")
    except OSError: pass
    print("Models are not downloaded or started by doctor.")
    return 0

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mega-transcribe", description="Local Termux audio transcription and coding assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect paths, dependencies and available storage")
    doctor.add_argument("--models-dir", type=Path, default=Path(os.environ.get("MEGA_MODELS_DIR", str(DEFAULT_MODELS_DIR))))
    tx = commands.add_parser("transcribe", help="Transcribe one file with a local NeMo model")
    tx.add_argument("audio", type=Path); tx.add_argument("--model", type=Path, default=None)
    tx.add_argument("--models-dir", type=Path, default=Path(os.environ.get("MEGA_MODELS_DIR", str(DEFAULT_MODELS_DIR))))
    tx.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    tx.add_argument("--language", default="ar"); tx.add_argument("--profile", choices=("faithful", "speech"), default="faithful")
    tx.add_argument("--formats", nargs="+", choices=("txt", "srt", "json"), default=("txt", "srt", "json"))
    assistant = commands.add_parser("assistant", help="Ask a local Qwen GGUF model")
    assistant.add_argument("--model", type=Path, default=_default_qwen()); assistant.add_argument("--prompt")
    assistant.add_argument("--context-size", type=int, default=1024); assistant.add_argument("--threads", type=int, default=2)
    commands.add_parser("menu", help="Open interactive terminal workflow")
    return parser

def _menu() -> int:
    print("Mega Transcribe — Termux\n1) Diagnostics\n2) Transcribe audio\n3) Start local web UI\n4) Ask Qwen\n0) Exit")
    choice = input("Choose: ").strip()
    if choice == "0": return 0
    if choice == "1": return _doctor(Path(os.environ.get("MEGA_MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser())
    if choice == "2":
        audio = Path(input("Audio file path: ").strip().strip('"')).expanduser()
        model = _default_nemo(Path(os.environ.get("MEGA_MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser())
        profile = input("Profile [faithful/speech] (faithful): ").strip() or "faithful"
        if profile not in {"faithful", "speech"}: print("Invalid profile"); return 2
        from .termux_pipeline import transcribe
        for path in transcribe(audio, model, PROJECT_ROOT / "outputs", profile=profile): print(f"Saved: {path}")
        return 0
    if choice == "3":
        from .web_app import main as web_main
        return web_main([])
    if choice == "4":
        prompt = input("Question: ").strip()
        from .assistant import ask
        print(ask(_default_qwen(), prompt, context_size=1024, threads=2)); return 0
    print("Unknown selection"); return 2

def main(argv: list[str] | None = None) -> int:
    parser = _parser(); args = parser.parse_args(argv)
    try:
        if args.command == "doctor": return _doctor(args.models_dir.expanduser().resolve())
        if args.command == "menu": return _menu()
        if args.command == "transcribe":
            from .termux_pipeline import transcribe
            model = args.model or _default_nemo(args.models_dir)
            for path in transcribe(args.audio, model, args.output_dir, language=args.language, profile=args.profile, formats=tuple(args.formats)): print(f"Saved: {path}")
            return 0
        if args.command == "assistant":
            if args.context_size < 512 or args.threads < 1: parser.error("--context-size must be >=512 and --threads >=1")
            from .assistant import ask
            if args.prompt: print(ask(args.model, args.prompt, context_size=args.context_size, threads=args.threads)); return 0
            while True:
                prompt = input("\nYou> ").strip()
                if prompt.lower() in {"/exit", "/quit"}: break
                if prompt: print("\nQwen> " + ask(args.model, prompt, context_size=args.context_size, threads=args.threads))
            return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr); return 1
    return 2
