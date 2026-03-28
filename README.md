# 🧠 Local LLM Medical Assistant (RAG Pipeline)

A fully local **AI-powered medical assistant** built using open-source LLMs, **Retrieval-Augmented Generation (RAG)**, and an optimized FastAPI backend.

## What this provides

- Local LLM via **Ollama** (e.g. `llama3`, `mistral`)
- PDF/Text ingestion → chunk → embed (**Sentence Transformers**) → index (**FAISS**)
- `POST /query` RAG answers grounded in retrieved context
- `POST /query-stream` **SSE streaming** token output (used by the web UI)
- `POST /upload-doc` upload PDFs / text files
- Background ingestion jobs (`async_mode=true`) + `GET /jobs/{id}`
- Prometheus metrics: `GET /metrics`
- Minimal modern web UI: `GET /`
- Fully local (no external LLM APIs)

## Safety note

This project is for **education/demo**. It is **not** a medical device and does not replace professional medical advice. The API responses intentionally include a medical safety disclaimer.

## Requirements

- Python 3.10+
- Ollama installed and running (default: `http://localhost:11434`)
  - Install: https://ollama.com
  - Run a model once so it’s available locally, e.g.: `ollama run llama3`

## Quickstart (Python)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-ml.txt -r requirements-dev.txt

# In another terminal, ensure Ollama is running and a model is pulled:
#   ollama run llama3

uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000/docs
UI: http://127.0.0.1:8000/

## API

### Upload documents

`POST /upload-doc`

- `file`: PDF or text
- Optional query params:
  - `source`: label stored with chunks
  - `async_mode=true`: returns a background `job_id` you can poll at `GET /jobs/{job_id}`

Example:

```powershell
curl -F "file=@.\docs\paper.pdf" "http://127.0.0.1:8000/upload-doc?source=paper"
```

### Query

`POST /query`

Body:

```json
{ "question": "What are the common side effects of metformin?" }
```

Optional fields:

- `top_k` (default 4)
- `model` (default from env `OLLAMA_MODEL`, e.g. `llama3`)

### Streaming query (SSE)

`POST /query-stream` returns `text/event-stream` with events:

- `sources`: JSON array of retrieved chunks (S1..Sn in order)
- `token`: streamed tokens as JSON string
- `done`: `true`

### OpenAI-style endpoint (for vLLM + SDK compatibility)

`POST /v1/chat/completions` accepts OpenAI-style `messages` and returns a standard chat completion.

- Uses the last `role="user"` message as the RAG question
- Supports `stream=true` (OpenAI SSE format with `data: ...` and final `[DONE]`)
- Supports `top_k` (non-standard field) for retrieval depth

Example:

```powershell
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d "{\"question\":\"What is hypertension?\"}"
```

## Persistence

Indexes and metadata are stored under `./data/` by default:

- `data/faiss.index`
- `data/chunks.jsonl`

You can change locations via environment variables (see `app/config.py`).

## Docker (optional)

This repo includes a `docker-compose.yml` that can run:

- `api` (FastAPI)
- `ollama` (Ollama server)
- `gateway` (Node.js gateway)

```bash
docker-compose up --build
```

Then browse:

- Gateway UI: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`

## vLLM / GPU mode (recruiter-grade)

If you have a GPU machine with NVIDIA Container Toolkit installed, you can run vLLM (OpenAI-compatible) for high concurrency:

```bash
docker compose -f docker-compose.gpu.yml up --build
```

This runs:

- `vllm` on `http://localhost:8001/v1`
- `api` configured with `LLM_BACKEND=openai_compat`
- `gateway` UI on `http://localhost:3000`

## Environment variables

- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (default `llama3`)
- `LLM_BACKEND` (`ollama` or `openai_compat`)
- `OPENAI_COMPAT_BASE_URL` (default `http://localhost:8001/v1`)
- `OPENAI_COMPAT_MODEL` (default `llama3`)
- `EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`)
- `DATA_DIR` (default `data`)
- `CHUNK_SIZE` (default `900`)
- `CHUNK_OVERLAP` (default `150`)
- `MAX_UPLOAD_MB` (default `25`)

## Benchmark

After ingesting a few documents:

```powershell
python scripts\bench.py --url http://127.0.0.1:8000 --requests 50 --concurrency 10
```

OpenAI-style benchmark:

```powershell
python scripts\bench.py --endpoint openai --url http://127.0.0.1:8000 --requests 50 --concurrency 10
python scripts\bench.py --endpoint openai --stream --url http://127.0.0.1:8000 --requests 20 --concurrency 5
```
