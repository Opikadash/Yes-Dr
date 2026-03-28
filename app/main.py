from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .doc_loader import load_text_from_bytes
from .jobs import JobStore
from .llm_backends import OllamaBackend, OpenAICompatBackend
from .metrics import (
    OLLAMA_GENERATION_DURATION_SECONDS,
    RAG_INGEST_CHUNKS_TOTAL,
    RAG_QUERIES_TOTAL,
    RAG_RETRIEVAL_DURATION_SECONDS,
    render_metrics,
    timer,
)
from .middleware import request_logging_middleware
from .ollama_client import OllamaClient
from .prompting import build_prompt, format_sources_for_prompt
from .rag_store import RAGStore
from .text_utils import chunk_text, join_context

app = FastAPI(title="Yes Doctor - Medical Assistant", version="0.1.0")
logging.basicConfig(level=logging.INFO)
app.middleware("http")(request_logging_middleware)

store = RAGStore(
    embedding_model_name=settings.embedding_model,
    index_path=settings.resolved_faiss_index_path(),
    chunks_path=settings.resolved_chunks_path(),
    docs_path=settings.resolved_docs_path(),
)

if settings.llm_backend.lower() == "openai_compat":
    llm = OpenAICompatBackend(
        base_url=settings.openai_compat_base_url,
        api_key=settings.openai_compat_api_key,
    )
    default_model = settings.openai_compat_model
else:
    llm = OllamaBackend(client=OllamaClient(base_url=settings.ollama_base_url))
    default_model = settings.ollama_model
jobs = JobStore()

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    model: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    top_k: int | None = Field(default=None, ge=1, le=20)


class UploadResponse(BaseModel):
    added_chunks: int
    doc_id: str | None = None
    skipped: bool = False
    reason: str | None = None
    sha256: str | None = None

class UploadJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    sha256: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    index = static_dir / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not packaged.")
    return FileResponse(str(index))


@app.get("/metrics")
def metrics():
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@app.get("/api/info")
def api_info():
    return {
        "llm_backend": settings.llm_backend,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
        "openai_compat_base_url": settings.openai_compat_base_url,
        "openai_compat_model": settings.openai_compat_model,
        "embedding_model": settings.embedding_model,
        "store": store.stats(),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "result": job.result,
        "error": job.error,
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _ingest_bytes(*, filename: str, content: bytes, source: str | None, force: bool) -> UploadResponse:
    text = load_text_from_bytes(filename, content)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")

    chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    sha = _sha256(content)
    result = store.add_document(
        filename=filename,
        sha256=sha,
        texts=chunks,
        source=source or filename,
        meta={"filename": filename},
        n_chars=len(text),
        created_at=_now_utc_iso(),
        force=force,
    )
    added = int(result.get("added_chunks", 0))
    RAG_INGEST_CHUNKS_TOTAL.inc(added)
    return UploadResponse(
        added_chunks=added,
        doc_id=result.get("doc_id"),
        skipped=bool(result.get("skipped", False)),
        reason=result.get("reason"),
        sha256=sha,
    )


@app.post("/upload-doc", response_model=UploadResponse | UploadJobResponse)
async def upload_doc(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    source: str | None = None,
    async_mode: bool = False,
    force: bool = False,
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.max_upload_mb}MB).")

    filename = file.filename or "upload"
    if not async_mode:
        return _ingest_bytes(filename=filename, content=content, source=source, force=force)

    job = jobs.create(kind="ingest")

    def run():
        jobs.set_running(job.id)
        try:
            result = _ingest_bytes(filename=filename, content=content, source=source, force=force).model_dump()
            jobs.set_done(job.id, result)
        except Exception as e:
            jobs.set_error(job.id, str(e))

    background.add_task(run)
    return UploadJobResponse(job_id=job.id, status="queued", sha256=_sha256(content))


@app.post("/upload-docs")
async def upload_docs(
    files: list[UploadFile] = File(...),
    source: str | None = None,
    force: bool = False,
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    results: list[dict] = []
    for f in files:
        content = await f.read()
        if not content:
            continue
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            continue
        filename = f.filename or "upload"
        results.append(_ingest_bytes(filename=filename, content=content, source=source, force=force).model_dump())
    return {"results": results}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    top_k = req.top_k or settings.default_top_k
    with timer(RAG_RETRIEVAL_DURATION_SECONDS):
        retrieved = store.search(query=req.question, top_k=top_k)
    context = join_context(format_sources_for_prompt(retrieved), max_chars=settings.max_context_chars)

    prompt = build_prompt(question=req.question, context=context)
    model = (req.model or default_model).strip()
    if not model:
        raise HTTPException(status_code=400, detail="No model specified.")

    try:
        RAG_QUERIES_TOTAL.labels("sync").inc()
        with timer(OLLAMA_GENERATION_DURATION_SECONDS, "sync"):
            answer = llm.generate(model=model, prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {e}")

    sources = [{"id": c.id, "source": c.source, "preview": c.text[:200]} for c in retrieved]
    return QueryResponse(answer=answer.strip(), sources=sources)


@app.post("/query-stream")
def query_stream(req: QueryRequest):
    top_k = req.top_k or settings.default_top_k
    with timer(RAG_RETRIEVAL_DURATION_SECONDS):
        retrieved = store.search(query=req.question, top_k=top_k)

    sources = [{"id": c.id, "source": c.source, "preview": c.text[:200]} for c in retrieved]
    context = join_context(format_sources_for_prompt(retrieved), max_chars=settings.max_context_chars)
    prompt = build_prompt(question=req.question, context=context)
    model = (req.model or default_model).strip()
    if not model:
        raise HTTPException(status_code=400, detail="No model specified.")

    def sse(event: str, data: str) -> str:
        return f"event: {event}\n" + f"data: {data}\n\n"

    def gen():
        yield sse("sources", json.dumps(sources, ensure_ascii=False))
        RAG_QUERIES_TOTAL.labels("stream").inc()
        with timer(OLLAMA_GENERATION_DURATION_SECONDS, "stream"):
            for token in llm.generate_stream(model=model, prompt=prompt):
                yield sse("token", json.dumps(token, ensure_ascii=False))
        yield sse("done", "true")

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/v1/models")
def v1_models():
    return {"object": "list", "data": [{"id": default_model, "object": "model", "created": 0, "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def v1_chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    # Use the last user message as the RAG question.
    question = ""
    for m in reversed(req.messages):
        if m.role == "user":
            question = m.content
            break
    if not question.strip():
        raise HTTPException(status_code=400, detail="No user message found.")

    top_k = req.top_k or settings.default_top_k
    with timer(RAG_RETRIEVAL_DURATION_SECONDS):
        retrieved = store.search(query=question, top_k=top_k)

    sources = [{"id": c.id, "source": c.source, "preview": c.text[:200]} for c in retrieved]
    context = join_context(format_sources_for_prompt(retrieved), max_chars=settings.max_context_chars)
    prompt = build_prompt(question=question, context=context)
    model = (req.model or default_model).strip()
    if not model:
        raise HTTPException(status_code=400, detail="No model specified.")

    created = int(time.time())
    completion_id = f"chatcmpl-{created}"

    if not req.stream:
        try:
            RAG_QUERIES_TOTAL.labels("sync").inc()
            with timer(OLLAMA_GENERATION_DURATION_SECONDS, "sync"):
                answer = llm.generate(model=model, prompt=prompt).strip()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM backend error: {e}")

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "rag_sources": sources,
        }

    def sse_data(obj) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def gen():
        # Initial chunk with assistant role (OpenAI streaming style)
        yield sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )

        RAG_QUERIES_TOTAL.labels("stream").inc()
        with timer(OLLAMA_GENERATION_DURATION_SECONDS, "stream"):
            for token in llm.generate_stream(model=model, prompt=prompt):
                yield sse_data(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                    }
                )

        yield sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
