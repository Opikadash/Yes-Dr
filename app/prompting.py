from __future__ import annotations

from typing import Iterable


def build_prompt(*, question: str, context: str) -> str:
    return (
        "You are a medical information assistant.\n"
        "Use ONLY the provided context. If the context is insufficient, say so.\n"
        "Cite facts with bracketed source ids like [S1], [S2].\n"
        "Keep the answer concise and structured (bullets are fine).\n"
        "Always include a short safety note advising consulting a qualified clinician.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:\n"
    )


def format_sources_for_prompt(chunks: Iterable) -> list[str]:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        label = f"S{i}"
        src = getattr(c, "source", None) or "unknown"
        cid = getattr(c, "id", None)
        text = getattr(c, "text", "")
        blocks.append(f"[{label}] source={src} chunk_id={cid}\n{text}")
    return blocks

