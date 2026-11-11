"""Optional local Qwen GGUF coding assistant; never executes or writes code."""

from __future__ import annotations

from pathlib import Path

SYSTEM_PROMPT = (
    "You are a local offline programming assistant. Explain code and suggest fixes. "
    "Do not claim to have executed code. Never execute commands or modify files. "
    "Make uncertainty and assumptions clear."
)


def ask(model_path: Path, prompt: str, *, context_size: int = 2048, threads: int = 2) -> str:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"ملف نموذج GGUF غير موجود: {model_path}")
    if not prompt.strip():
        raise ValueError("اكتب سؤالًا أو وصف الخطأ أولًا.")
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python غير مثبت. تثبيته وبناؤه على ARM64 قد يحتاج مساحة وأدوات بناء؛ راجع README.md."
        ) from exc

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=context_size,
            n_threads=threads,
            n_batch=min(128, context_size),
            verbose=False,
        )
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=min(1024, context_size // 2),
        )
    except Exception as exc:
        raise RuntimeError(f"فشل تشغيل نموذج GGUF ({type(exc).__name__}): {exc}") from exc
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError("أعاد النموذج استجابة فارغة.")
    return str(choices[0].get("message", {}).get("content") or "").strip()
