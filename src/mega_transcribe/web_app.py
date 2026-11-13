"""Small local-only HTTP interface for Termux. Bind to loopback by default."""
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

ROOT = Path(os.environ.get("MEGA_PROJECT_DIR", str(Path(__file__).resolve().parents[2]))).expanduser().resolve()
MODELS_DIR = Path(os.environ.get("MEGA_MODELS_DIR", str(DEFAULT_MODELS_DIR))).expanduser()
OUTPUT_DIR = Path(os.environ.get("MEGA_OUTPUT_DIR", str(ROOT / "outputs"))).expanduser()
UPLOAD_DIR = OUTPUT_DIR / ".uploads"
MAX_UPLOAD = 48 * 1024 * 1024
MAX_JOBS = 40
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
WORK: queue.Queue[tuple[str, Path, str | None, str, tuple[str, ...], float, float]] = queue.Queue(maxsize=2)
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".amr", ".ogg", ".opus", ".flac", ".3gp", ".mp4", ".webm"}


def _worker() -> None:
    while True:
        job_id, audio_path, language, profile, formats, denoise, gain_db = WORK.get()
        try:
            with LOCK:
                JOBS[job_id].update(status="processing", message="Preparing temporary PCM and running local NeMo ASR.")
            model_path = _default_nemo(MODELS_DIR)
            outputs = transcribe(audio_path, model_path, OUTPUT_DIR, language=language, profile=profile, formats=formats, denoise_strength=denoise, gain_db=gain_db)
            with LOCK:
                JOBS[job_id].update(status="complete", message="Completed. Check JSON for actual timestamp precision.", outputs=[{"name": p.name, "url": f"/outputs/{p.name}"} for p in outputs])
        except Exception as exc:
            with LOCK:
                if job_id in JOBS:
                    JOBS[job_id].update(status="error", message=str(exc))
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
            WORK.task_done()


threading.Thread(target=_worker, daemon=True, name="mega-audio-worker").start()


class Handler(BaseHTTPRequestHandler):
    server_version = "MegaTranscribe/0.4"
    sys_version = ""

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _local_origin(self) -> bool:
        origin = self.headers.get("Origin")
        return not origin or urlparse(origin).hostname in {"127.0.0.1", "localhost", "::1"}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            page = Path(__file__).with_name("web").joinpath("index.html")
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        elif parsed.path == "/api/health":
            nemo = _default_nemo(MODELS_DIR)
            self._json(200, {"ok": True, "models_dir": str(MODELS_DIR), "nemo_found": nemo.is_file(), "nemo_model": str(nemo), "qwen_found": _default_qwen().is_file()})
        elif parsed.path == "/api/models":
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
            self._json(200, {"models": models, "qwen": str(qwen) if qwen.is_file() else None, "notice": "Non-NeMo assets are inventoried only until their matching inference backend is configured."})
        elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}", parsed.path):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with LOCK:
                job = JOBS.get(job_id)
            self._json(200, dict(job) if job else {"error": "job not found"})
        elif parsed.path.startswith("/outputs/"):
            raw_name = unquote(parsed.path.removeprefix("/outputs/"))
            name = Path(raw_name).name
            target = (OUTPUT_DIR / name).resolve()
            if name != raw_name or name.startswith(".") or target.parent != OUTPUT_DIR.resolve() or not target.is_file():
                self._json(404, {"error": "output not found"})
            else:
                ctype = "application/x-subrip; charset=utf-8" if target.suffix == ".srt" else "application/json; charset=utf-8" if target.suffix == ".json" else "text/plain; charset=utf-8"
                self._send(200, target.read_bytes(), ctype)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._local_origin():
            self._json(403, {"error": "non-local browser origin rejected"})
        elif self.path == "/api/transcribe":
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
            self._json(413, {"error": f"upload limit is {MAX_UPLOAD // (1024 * 1024)} MiB"}); return
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json(400, {"error": "expected multipart/form-data"}); return
        body = self.rfile.read(length)
        message = BytesParser(policy=policy.default).parsebytes(f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body)
        if not message.is_multipart():
            self._json(400, {"error": "invalid multipart body"}); return
        file_bytes = None
        filename = ""
        fields: dict[str, str] = {}
        for part in message.iter_parts():
            field = part.get_param("name", header="content-disposition")
            if field == "audio":
                filename = Path(part.get_filename() or "audio").name
                file_bytes = part.get_payload(decode=True)
            elif field:
                fields[str(field)] = (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace")[:200]
        if not file_bytes or Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            self._json(400, {"error": "choose supported audio: WAV, MP3, M4A, AMR, OGG, FLAC, 3GP, etc."}); return
        path: Path | None = None
        try:
            profile = fields.get("profile", "faithful")
            if profile not in {"faithful", "speech"}:
                raise ValueError("invalid preprocessing profile")
            denoise = float(fields.get("denoise", "0"))
            gain_db = float(fields.get("gain_db", "0"))
            if not 0 <= denoise <= 1 or not -12 <= gain_db <= 12:
                raise ValueError("denoise must be 0..1 and gain must be -12..12 dB")
            requested = fields.get("formats", "txt,srt,json")
            formats = tuple(dict.fromkeys(x for x in requested.split(",") if x in {"txt", "srt", "json"}))
            if not formats:
                raise ValueError("choose at least one output format")
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:80] or "audio"
            path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
            path.write_bytes(file_bytes)
            job_id = uuid.uuid4().hex
            language = fields.get("language", "ar").strip() or None
            with LOCK:
                if len(JOBS) >= MAX_JOBS:
                    for old_id in list(JOBS):
                        if JOBS[old_id].get("status") in {"complete", "error"}:
                            del JOBS[old_id]
                            if len(JOBS) < MAX_JOBS:
                                break
                JOBS[job_id] = {"id": job_id, "status": "queued", "message": "Waiting for local worker", "outputs": []}
            WORK.put_nowait((job_id, path, language, profile, formats, denoise, gain_db))
        except queue.Full:
            if path:
                path.unlink(missing_ok=True)
            self._json(503, {"error": "processing queue is full; wait for the current job"}); return
        except (OSError, ValueError) as exc:
            if path:
                path.unlink(missing_ok=True)
            self._json(400, {"error": str(exc)}); return
        self._json(202, {"id": job_id, "status": "queued"})

    def _assistant(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 32 * 1024:
                self._json(413, {"error": "prompt request exceeds 32 KiB"}); return
            payload = json.loads(self.rfile.read(length))
            prompt = str(payload.get("prompt", ""))[:8000]
            model = Path(str(payload.get("model") or _default_qwen())).expanduser()
            if not prompt.strip():
                self._json(400, {"error": "enter a prompt"}); return
            from .assistant import ask
            answer = ask(model, prompt, context_size=1024, threads=2)
            self._json(200, {"answer": answer})
        except Exception as exc:
            self._json(500, {"error": str(exc)})


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Local-only lightweight Mega Transcribe UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("The UI only binds to loopback for safety.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Local UI: http://{args.host}:{args.port}")
    print(f"Model directory: {MODELS_DIR}")
    print("Audio and prompts stay on this device. Ctrl+C stops the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
