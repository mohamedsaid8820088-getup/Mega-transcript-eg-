"""Loopback-only local web UI for CTC, optional NeMo, and Pyannote raw segmentation."""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import uuid
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from .cli import _default_nemo
from .ctc_backend import default_assets, transcribe_ctc
from .model_inventory import inventory
from .pyannote_onnx import analyze_segmentation, default_segmentation_path
from .termux_pipeline import transcribe as transcribe_nemo

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("MEGA_MODELS_DIR", str(ROOT / "models" / "AudioModels"))).expanduser()
OUTPUT_DIR = Path(os.environ.get("MEGA_OUTPUT_DIR", str(ROOT / "outputs"))).expanduser()
UPLOAD_DIR = OUTPUT_DIR / ".uploads"
MAX_UPLOAD = 48 * 1024 * 1024
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
WORK: queue.Queue[tuple[str, Path, tuple[str, ...], int, tuple[str, ...], str]] = queue.Queue(maxsize=2)
ALLOWED = {".wav", ".mp3", ".m4a", ".aac", ".amr", ".ogg", ".opus", ".flac", ".3gp", ".mp4", ".webm"}


def _update(job_id: str, **values: object) -> None:
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)


def _worker() -> None:
    while True:
        job_id, audio, formats, threads, engines, profile = WORK.get()
        engine_results: list[dict] = []
        all_outputs: list[dict] = []
        errors: list[str] = []
        try:
            for index, engine in enumerate(engines):
                base = 5 + int(90 * index / len(engines))
                _update(job_id, status="processing", progress=base, stage="prepare", message=f"بدء {engine} ({index + 1}/{len(engines)})")
                engine_audio = UPLOAD_DIR / f"{job_id}_{engine}_{audio.name}"
                try:
                    if engine == "pyannote":
                        model = default_segmentation_path(MODELS_DIR)
                        destination = OUTPUT_DIR / f"{audio.stem}_{job_id}_pyannote-segmentation.json"
                        def report(percent: int, message: str) -> None:
                            overall = base + int((90 / len(engines)) * percent / 100)
                            _update(job_id, progress=min(98, overall), stage="inference", message=f"Pyannote: {message}")
                        outputs = [analyze_segmentation(audio, model, destination, threads=threads, progress=report)]
                    else:
                        shutil.copyfile(audio, engine_audio)
                        if engine == "ctc":
                            def report_ctc(percent: int, stage: str, message: str) -> None:
                                overall = base + int((90 / len(engines)) * percent / 100)
                                _update(job_id, progress=min(98, overall), stage=stage, message=f"CTC: {message}")
                            outputs = transcribe_ctc(engine_audio, OUTPUT_DIR, formats=formats, threads=threads, progress=report_ctc)
                        else:
                            model = _default_nemo(MODELS_DIR)
                            if not model.is_file():
                                raise FileNotFoundError(f"لم يُعثر على نموذج NeMo .nemo: {model}")
                            _update(job_id, progress=base + 5, stage="load_model", message="NeMo: تحميل النموذج المحلي على CPU")
                            outputs = transcribe_nemo(engine_audio, model, OUTPUT_DIR, language="ar", profile=profile, formats=formats)
                    entries = [{"engine": engine, "name": p.name, "url": f"/outputs/{quote(p.name)}"} for p in outputs]
                    all_outputs.extend(entries)
                    engine_results.append({"engine": engine, "status": "complete", "outputs": entries})
                except Exception as exc:
                    errors.append(f"{engine}: {exc}")
                    engine_results.append({"engine": engine, "status": "error", "error": str(exc)})
                finally:
                    engine_audio.unlink(missing_ok=True)
            if all_outputs:
                message = "اكتملت المعالجة المحلية." + (("\n" + "\n".join(errors)) if errors else "")
                _update(job_id, status="complete", progress=100, stage="complete", message=message, outputs=all_outputs, engines=engine_results)
            else:
                _update(job_id, status="error", stage="error", message="فشلت كل المحركات المحددة. " + "\n".join(errors), engines=engine_results)
        except Exception as exc:
            _update(job_id, status="error", stage="error", message=str(exc), engines=engine_results)
        finally:
            audio.unlink(missing_ok=True)
            WORK.task_done()


threading.Thread(target=_worker, daemon=True, name="mega-asr-worker").start()


class Handler(BaseHTTPRequestHandler):
    server_version = "MegaTranscribeLocal/1.3"
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
            self._send(200, Path(__file__).with_name("web").joinpath("ctc.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/health":
            try:
                model, vocab = default_assets(MODELS_DIR)
                ctc_ok, ctc_error = True, None
            except Exception as exc:
                model, vocab, ctc_ok, ctc_error = None, None, False, str(exc)
            nemo = _default_nemo(MODELS_DIR)
            try:
                import torch  # noqa: F401
                from nemo.collections.asr.models import ASRModel  # noqa: F401
                nemo_runtime, nemo_error = True, None
            except Exception as exc:
                nemo_runtime, nemo_error = False, f"{type(exc).__name__}: {exc}"
            segmentation = default_segmentation_path(MODELS_DIR)
            try:
                import onnxruntime  # noqa: F401
                onnx_available = True
            except Exception:
                onnx_available = False
            self._json(200, {"ok": ctc_ok, "model": str(model) if model else None, "vocab": str(vocab) if vocab else None, "ctc_error": ctc_error, "models_dir": str(MODELS_DIR), "nemo_model": str(nemo), "nemo_found": nemo.is_file(), "nemo_runtime": nemo_runtime, "nemo_error": nemo_error, "pyannote_segmentation": str(segmentation), "pyannote_found": segmentation.is_file(), "onnx_runtime_available": onnx_available})
        elif path == "/api/models":
            self._json(200, inventory(MODELS_DIR))
        elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}", path):
            with LOCK:
                job = JOBS.get(path.rsplit("/", 1)[-1])
            self._json(200, dict(job) if job else {"error": "الوظيفة غير موجودة"})
        elif path.startswith("/outputs/"):
            raw = unquote(path.removeprefix("/outputs/"))
            relative = Path(raw)
            target = (OUTPUT_DIR / relative).resolve()
            if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts) or relative.name.startswith(".") or OUTPUT_DIR.resolve() not in target.parents or not target.is_file():
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
            self._json(400, {"error": "اختر ملف صوت مدعومًا مثل WAV أو MP3 أو M4A أو AMR أو FLAC"})
            return
        formats = tuple(dict.fromkeys(x for x in fields.get("formats", "txt,srt,json").split(",") if x in {"txt", "srt", "json"}))
        if not formats:
            self._json(400, {"error": "اختر صيغة إخراج واحدة على الأقل"})
            return
        engines = tuple(dict.fromkeys(x for x in fields.get("engines", "ctc").split(",") if x in {"ctc", "nemo", "pyannote"}))
        if not engines:
            self._json(400, {"error": "اختر محركًا أو أداة واحدة على الأقل"})
            return
        try:
            threads = int(fields.get("threads", "2"))
        except ValueError:
            self._json(400, {"error": "قيمة خيوط CPU غير صالحة"})
            return
        if not 1 <= threads <= 4:
            self._json(400, {"error": "اختر من 1 إلى 4 خيوط CPU"})
            return
        profile = fields.get("profile", "faithful")
        if profile not in {"faithful", "speech"}:
            self._json(400, {"error": "إعداد معالجة الصوت غير صالح"})
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
            JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0, "stage": "queued", "message": "في انتظار المعالجة المحلية", "outputs": [], "engines": [], "settings": {"threads": threads, "engines": list(engines), "formats": list(formats), "profile": profile}}
        try:
            WORK.put_nowait((job_id, audio, formats, threads, engines, profile))
        except queue.Full:
            audio.unlink(missing_ok=True)
            with LOCK:
                JOBS.pop(job_id, None)
            self._json(503, {"error": "قائمة المعالجة ممتلئة؛ حاول بعد اكتمال المهمة الحالية"})
            return
        self._json(202, {"id": job_id, "status": "queued"})


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="واجهة محلية لـ CTC وPyannote segmentation وNeMo الاختياري")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("الواجهة تقبل اتصالات الجهاز المحلي فقط.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"واجهة التفريغ المحلية: http://{args.host}:{args.port}")
    print(f"مجلد النماذج: {MODELS_DIR}")
    print("الصوت يعالج محليًا؛ Ctrl+C لإيقاف الخادم.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nإيقاف الواجهة.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
