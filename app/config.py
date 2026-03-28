from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    llm_backend: str = "ollama"  # "ollama" | "openai_compat"
    openai_compat_base_url: str = "http://localhost:8001/v1"
    openai_compat_api_key: str = ""
    openai_compat_model: str = "llama3"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    data_dir: Path = Path("data")
    faiss_index_path: Path | None = None
    chunks_path: Path | None = None
    docs_path: Path | None = None

    chunk_size: int = 900
    chunk_overlap: int = 150

    default_top_k: int = 4
    max_context_chars: int = 8000
    max_upload_mb: int = 25

    def resolved_faiss_index_path(self) -> Path:
        if self.faiss_index_path is not None:
            return self.faiss_index_path
        return self.data_dir / "faiss.index"

    def resolved_chunks_path(self) -> Path:
        if self.chunks_path is not None:
            return self.chunks_path
        return self.data_dir / "chunks.jsonl"

    def resolved_docs_path(self) -> Path:
        if self.docs_path is not None:
            return self.docs_path
        return self.data_dir / "docs.jsonl"


settings = Settings()
