from __future__ import annotations

from pypdf import PdfReader

from .text_utils import normalize_text


def load_text_from_bytes(filename: str, content: bytes) -> str:
    name = filename.lower().strip()

    if name.endswith(".pdf"):
        reader = PdfReader(io_bytes_to_pathless_stream(content))
        parts: list[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return normalize_text("\n".join(parts))

    # Default: treat as UTF-8 text.
    try:
        return normalize_text(content.decode("utf-8", errors="ignore"))
    except Exception:
        return ""


def io_bytes_to_pathless_stream(content: bytes):
    # pypdf works with file-like objects.
    import io

    return io.BytesIO(content)
