"""Loopback-only local web UI for ONNX CTC transcription and model status."""
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

from .ctc_backend import default_assets, transcribe_ctc
from .model_inventory import inventory

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("MEGA_MODELS_DIR", str(ROOT / "models" / "AudioModels"))).expanduser()
OUTPUT_DIR = Path(os.environ.get("MEGA_OUTPUT_DIR", str(ROOT / "outputs"))).expanduser()
UPLOAD_DIR = OUTPUT_DIR / ".uploads"
MAX_UPLOAD = 48 * 1024 * 1024
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
WORK: queue.Queue[tuple[str, Path, tuple[str, ...], int]] = queue.Queue(maxsize=2)
ALLOWED = {".wav", ".mp3", ".m4a", ".aac", ".amr", ".ogg", ".opus", ".flac", ".3gp", ".mp4", ".webm"}


def _update_progress(job_id: str, percent: int, stage: str, message: str) -> None:
    with LOCK:
        job = JOBS.get(job_id)
        if job:
            job.update(progress=max(0, min(99, int(percent))), stage=stage, message=message)


def _worker() -> None:
    while True:
        job_id, audio, formats, threads = WORK.get()
        try:
            _update_progress(job_id, 5, "prepare", "بدأت المعالجة المحلية")
            outputs = transcribe_ctc(audio, OUTPUT_DIR, formats=formats, threads=threads, progress=lambda p, s, m: _update_progress(job_id, p, s, m))
            with LOCK:
                if job_id in JOBS:
                    JOBS[job_id].update(status="complete", progress=100, stage="complete", message="اكتمل التفريغ. راجع النص خصوصًا في المقاطع المشوشة.", outputs=[{"name": p.name, "url": f"/outputs/{p.name}"} for p in outputs])
        except Exception as exc:
            with LOCK:
                if job_id in JOBS:
                    JOBS[job_id].update(status="error", stage="error", message=str(exc))
        finally:
            try:
                audio.unlink(missing_ok=True)
            except OSError:
                pass
            WORK.task_done()


threading.Thread(target=_worker, daemon=True, name="mega-ctc-worker").start()


class Handler(BaseHTTPRequestHandler):
    server_version = "MegaTranscribeCTC/1.1"
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
        path = urlparse(self.path).path
        if path == "/":
            page = Path(__file__).with_name("web").joinpath("ctc.html")
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/health":
            try:
                model, vocab = default_assets(MODELS_DIR)
                self._json(200, {"ok": True, "backend": "ONNX CTC", "model": str(model), "vocab": str(vocab), "models_dir": str(MODELS_DIR)})
            except Exception as exc:
                self._json(200, {"ok": False, "backend": "ONNX CTC", "error": str(exc), "models_dir": str(MODELS_DIR)})
        elif path == "/api/models":
            try:
                self._json(200, inventory(MODELS_DIR))
            except OSError as exc:
                self._json(500, {"error": str(exc), "models": []})
        elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}", path):
            with LOCK:
                job = JOBS.get(path.rsplit("/", 1)[-1])
            self._json(200, dict(job) if job else {"error": "الوظيفة غير موجودة"})
        elif path.startswith("/outputs/"):
            raw = unquote(path.removeprefix("/outputs/"))
            name = Path(raw).name
            target = (OUTPUT_DIR / name).resolve()
            if name != raw or name.startswith(".") or target.parent != OUTPUT_DIR.resolve() or not target.is_file():
                self._json(404, {"error": "الملف غير موجود"})
            else:
                ctype = "application/x-subrip; charset=utf-8" if target.suffix == ".srt" else "application/json; charset=utf-8" if target.suffix == ".json" else "text/plain; charset=utf-8"
                self._send(200, target.read_bytes(), ctype)
        else:
            self._json(404, {"error": "غير موجود"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._local_origin():
            self._json(403, {"error": "تم رفض مصدر متصفح غير محلي"})
            return
        if self.path != "/api/transcribe":
            self._json(404, {"error": "غير موجود"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "طول الطلب غير صالح"})
            return
        if length <= 0 or length > MAX_UPLOAD:
            self._json(413, {"error": "الحد الأقصى للملف 48 ميبيبايت"})
            return
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json(400, {"error": "الطلب يجب أن يكون multipart/form-data"})
            return
        body = self.rfile.read(length)
        message = BytesParser(policy=policy.default).parsebytes(f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body)
        audio_data, filename, fields = None, "audio", {}
        if message.is_multipart():
            for part in message.iter_parts():
                field = part.get_param("name", header="content-disposition")
                if field == "audio":
                    filename = Path(part.get_filename() or "audio").name
                    audio_data = part.get_payload(decode=True)
                elif field:
                    fields[str(field)] = (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace")[:200]
        if not audio_data or Path(filename).suffix.lower() not in ALLOWED:
            self._json(400, {"error": "اختر ملفًا صوتيًا مدعومًا مثل WAV أو MP3 أو M4A أو AMR أو FLAC"})
            return
        formats = tuple(dict.fromkeys(x for x in fields.get("formats", "txt,srt,json").split(",") if x in {"txt", "srt", "json"}))
        if not formats:
            self._json(400, {"error": "اختر صيغة إخراج واحدة على الأقل"})
            return
        try:
            threads = int(fields.get("threads", "2"))
        except ValueError:
            self._json(400, {"error": "قيمة خيوط CPU غير صالحة"})
            return
        if not 1 <= threads <= 4:
            self._json(400, {"error": "اختر من 1 إلى 4 خيوط CPU"})
            return
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:80] or "audio.wav"
        audio = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe}"
        try:
            audio.write_bytes(audio_data)
        except OSError as exc:
            self._json(500, {"error": f"تعذر حفظ الرفع المؤقت: {exc}"})
            return
        job_id = uuid.uuid4().hex
        with LOCK:
            JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0, "stage": "queued", "message": "في انتظار المعالجة المحلية", "outputs": [], "settings": {"cpu_threads": threads, "formats": list(formats)}}
        try:
            WORK.put_nowait((job_id, audio, formats, threads))
        except queue.Full:
            audio.unlink(missing_ok=True)
            with LOCK:
                JOBS.pop(job_id, None)
            self._json(503, {"error": "قائمة المعالجة ممتلئة؛ حاول بعد اكتمال المهمة الحالية"})
            return
        self._json(202, {"id": job_id, "status": "queued"})


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="واجهة التفريغ المحلية بنموذج CTC")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("الواجهة تقبل اتصالات الجهاز المحلي فقط.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"واجهة التفريغ المحلية: http://{args.host}:{args.port}")
    print(f"مجلد النماذج: {MODELS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nإيقاف الواجهة.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
