from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "HTTP requests total",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

RAG_INGEST_CHUNKS_TOTAL = Counter(
    "rag_ingest_chunks_total",
    "Total chunks ingested",
)

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "RAG queries total",
    ["mode"],  # "sync" | "stream"
)

RAG_RETRIEVAL_DURATION_SECONDS = Histogram(
    "rag_retrieval_duration_seconds",
    "Vector retrieval duration in seconds",
)

OLLAMA_GENERATION_DURATION_SECONDS = Histogram(
    "ollama_generation_duration_seconds",
    "Ollama generation duration in seconds",
    ["mode"],  # "sync" | "stream"
)


@contextmanager
def timer(hist: Histogram, *labels):
    start = time.perf_counter()
    try:
        yield
    finally:
        hist.labels(*labels).observe(time.perf_counter() - start) if labels else hist.observe(time.perf_counter() - start)


def render_metrics():
    return generate_latest(), CONTENT_TYPE_LATEST

