from __future__ import annotations

import re
from typing import Iterable


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    # Simple character-based chunking (fast, dependency-free).
    # Overlap helps preserve context across boundaries.
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def join_context(chunks: Iterable[str], *, max_chars: int) -> str:
    sep = "\n\n---\n\n"
    out: list[str] = []
    total = 0
    for c in chunks:
        if not c:
            continue

        sep_len = len(sep) if out else 0
        remaining = max_chars - total
        if remaining <= 0:
            break
        if sep_len and remaining <= sep_len:
            break

        remaining_after_sep = remaining - sep_len
        if remaining_after_sep <= 0:
            break

        piece = c if len(c) <= remaining_after_sep else c[:remaining_after_sep]
        if not piece:
            break

        out.append(piece)
        total += sep_len + len(piece)

    return sep.join(out)
