"""Small local-only HTTP interface for Termux."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import uuid
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .cli import DEFAULT_MODELS_DIR, _default_nemo, _default_qwen
from .termux_pipeline import transcribe

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("MEGA_MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser()
OUTPUT_DIR = Path(os.environ.get("MEGA_OUTPUT_DIR", str(ROOT / "outputs"))).expanduser()
UPLOAD_DIR = OUTPUT_DIR / ".uploads"
MAX_UPLOAD = 200 * 1024 * 1024
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
WORK: queue.Queue[tuple[str, Path, str | None]] = queue.Queue()
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".amr", ".ogg", ".opus", ".flac", ".3gp", ".mp4", ".webm"}


def _worker() -> None:
    while True:
        job_id, audio_path, language = WORK.get()
        try:
            with LOCK:
                JOBS[job_id]["status"] = "processing"
                JOBS[job_id]["message"] = "Converting a temporary copy to mono 16 kHz PCM and running local NeMo ASR."
            model_path = _default_nemo(MODELS_DIR)
            outputs = transcribe(audio_path, model_path, OUTPUT_DIR, language=language)
            with LOCK:
                JOBS[job_id].update(status="complete", message="Done", outputs=[{"name": p.name, "url": f"/outputs/{p.name}"} for p in outputs])
        except Exception as exc:
            with LOCK:
                JOBS[job_id].update(status="error", message=str(exc))
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
            WORK.task_done()


threading.Thread(target=_worker, daemon=True, name="mega-audio-worker").start()


class Handler(BaseHTTPRequestHandler):
    server_version = "MegaTranscribe/0.2"

    def log_message(self, fmt: str, *args: object) -> None:
        # Do not log request bodies, audio contents, or prompts.
        super().log_message(fmt, *args)

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            page = Path(__file__).with_name("web").joinpath("index.html")
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/health":
            nemo = _default_nemo(MODELS_DIR)
            self._json(200, {"ok": True, "models_dir": str(MODELS_DIR), "nemo_found": nemo.is_file(), "nemo_model": str(nemo), "qwen_found": _default_qwen().is_file()})
            return
        if parsed.path == "/api/models":
            models = []
            if MODELS_DIR.exists():
                for path in MODELS_DIR.rglob("*"):
                    if path.is_file() and path.suffix.lower() in {".nemo", ".onnx", ".gguf", ".safetensors"}:
                        try:
                            models.append({"name": str(path.relative_to(MODELS_DIR)), "type": path.suffix.lower(), "bytes": path.stat().st_size})
                        except OSError:
                            continue
                        if len(models) >= 300:
                            break
            qwen = _default_qwen()
            self._json(200, {"models": models, "qwen": str(qwen) if qwen.is_file() else None,
                             "notice": "ONNX/CTC/Pyannote/MossFormer artifacts are inventoried only until their matching runtime/config is verified."})
            return
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})", parsed.path)
        if match:
            with LOCK:
                job = JOBS.get(match.group(1))
            if job is None:
                self._json(404, {"error": "job not found"})
            else:
                self._json(200, dict(job))
            return
        if parsed.path.startswith("/outputs/"):
            name = Path(unquote(parsed.path.removeprefix("/outputs/"))).name
            if name != unquote(parsed.path.removeprefix("/outputs/")) or name.startswith("."):
                self._json(400, {"error": "invalid output path"})
                return
            target = (OUTPUT_DIR / name).resolve()
            if target.parent != OUTPUT_DIR.resolve() or not target.is_file():
                self._json(404, {"error": "output not found"})
                return
            content_type = "text/plain; charset=utf-8" if target.suffix == ".txt" else "application/json; charset=utf-8" if target.suffix == ".json" else "application/x-subrip; charset=utf-8"
            self._send(200, target.read_bytes(), content_type)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/transcribe":
            self._upload()
        elif self.path == "/api/assistant":
            self._assistant()
        else:
            self._json(404, {"error": "not found"})

    def _upload(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "invalid content length"}); return
        if length <= 0 or length > MAX_UPLOAD:
            self._json(413, {"error": f"upload must be between 1 byte and {MAX_UPLOAD // (1024*1024)} MiB"}); return
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json(400, {"error": "expected multipart/form-data"}); return
        body = self.rfile.read(length)
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
        )
        file_bytes = None
        filename = ""
        fields: dict[str, str] = {}
        if not message.is_multipart():
            self._json(400, {"error": "invalid multipart body"}); return
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "audio":
                filename = Path(part.get_filename() or "audio").name
                file_bytes = part.get_payload(decode=True)
            elif name:
                raw = part.get_payload(decode=True) or b""
                fields[str(name)] = raw.decode("utf-8", errors="replace")[:200]
        if not file_bytes or Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            self._json(400, {"error": "select a supported audio file (wav, mp3, m4a, amr, ogg, flac, 3gp, etc.)"}); return
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:120] or "audio.bin"
        path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
        path.write_bytes(file_bytes)
        job_id = uuid.uuid4().hex
        language = fields.get("language", "ar").strip() or None
        with LOCK:
            JOBS[job_id] = {"id": job_id, "status": "queued", "message": "Queued", "outputs": []}
        WORK.put((job_id, path, language))
        self._json(202, {"id": job_id, "status": "queued"})

    def _assistant(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                self._json(413, {"error": "prompt request too large"}); return
            payload = json.loads(self.rfile.read(length))
            prompt = str(payload.get("prompt", ""))[:12000]
            model = Path(str(payload.get("model") or _default_qwen())).expanduser()
            if not prompt.strip():
                self._json(400, {"error": "enter a prompt"}); return
            from .assistant import ask
            answer = ask(model, prompt, context_size=1024, threads=2)
            self._json(200, {"answer": answer})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def do_HEAD(self) -> None:  # noqa: N802
        self._json(405, {"error": "method not allowed"})


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Local-only lightweight Mega Transcribe web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Keep localhost unless you explicitly need LAN access")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Local UI: http://{args.host}:{args.port}")
    print(f"Models: {MODELS_DIR}")
    print("Audio and prompts stay on this device. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
