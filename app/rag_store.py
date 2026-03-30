from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class StoredChunk:
    id: str
    text: str
    doc_id: str | None = None
    source: str | None = None
    meta: dict[str, Any] | None = None


@dataclass
class StoredDoc:
    doc_id: str
    filename: str
    sha256: str
    source: str | None = None
    n_chars: int | None = None
    n_chunks: int | None = None
    created_at: str | None = None


class RAGStore:
    def __init__(
        self,
        *,
        embedding_model_name: str,
        index_path: Path,
        chunks_path: Path,
        docs_path: Path,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.docs_path = docs_path

        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._chunks: list[StoredChunk] = []
        self._docs: list[StoredDoc] = []
        self._doc_hash_to_id: dict[str, str] = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    @property
    def dim(self) -> int:
        # SentenceTransformer exposes embedding dimension in several ways across versions.
        # The following is stable for v3+.
        return int(self.model.get_sentence_embedding_dimension())

    def _ensure_index(self) -> faiss.Index:
        if self._index is not None:
            return self._index
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        self.docs_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and self.chunks_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._chunks = list(self._load_chunks())
            self._docs = list(self._load_docs())
            self._doc_hash_to_id = {d.sha256: d.doc_id for d in self._docs if d.sha256}
            return self._index

        self._index = faiss.IndexFlatIP(self.dim)
        self._chunks = []
        self._docs = []
        self._doc_hash_to_id = {}
        return self._index

    def _load_chunks(self) -> list[StoredChunk]:
        chunks: list[StoredChunk] = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(
                    StoredChunk(
                        id=str(obj["id"]),
                        text=str(obj["text"]),
                        doc_id=obj.get("doc_id"),
                        source=obj.get("source"),
                        meta=obj.get("meta"),
                    )
                )
        return chunks

    def _load_docs(self) -> list[StoredDoc]:
        if not self.docs_path.exists():
            return []
        docs: list[StoredDoc] = []
        with self.docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                docs.append(
                    StoredDoc(
                        doc_id=str(obj["doc_id"]),
                        filename=str(obj.get("filename", "")),
                        sha256=str(obj.get("sha256", "")),
                        source=obj.get("source"),
                        n_chars=obj.get("n_chars"),
                        n_chunks=obj.get("n_chunks"),
                        created_at=obj.get("created_at"),
                    )
                )
        return docs

    def _append_chunks(self, new_chunks: list[StoredChunk]) -> None:
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with self.chunks_path.open("a", encoding="utf-8") as f:
            for c in new_chunks:
                f.write(
                    json.dumps(
                        {"id": c.id, "text": c.text, "doc_id": c.doc_id, "source": c.source, "meta": c.meta},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _append_doc(self, doc: StoredDoc) -> None:
        self.docs_path.parent.mkdir(parents=True, exist_ok=True)
        with self.docs_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "filename": doc.filename,
                        "sha256": doc.sha256,
                        "source": doc.source,
                        "n_chars": doc.n_chars,
                        "n_chunks": doc.n_chunks,
                        "created_at": doc.created_at,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return emb

    def stats(self) -> dict[str, Any]:
        index = self._ensure_index()
        return {"vectors": int(index.ntotal), "chunks": len(self._chunks), "docs": len(self._docs)}

    def find_doc_by_hash(self, sha256: str) -> StoredDoc | None:
        sha256 = sha256.strip().lower()
        if not sha256:
            return None
        self._ensure_index()
        doc_id = self._doc_hash_to_id.get(sha256)
        if not doc_id:
            return None
        for d in self._docs:
            if d.doc_id == doc_id:
                return d
        return None

    def add_document(
        self,
        *,
        filename: str,
        sha256: str,
        texts: list[str],
        source: str | None = None,
        meta: dict[str, Any] | None = None,
        n_chars: int | None = None,
        created_at: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        if not texts:
            return {"doc_id": None, "added_chunks": 0, "skipped": True, "reason": "empty"}
        sha256 = sha256.strip().lower()

        existing = self.find_doc_by_hash(sha256) if sha256 else None
        if existing and not force:
            return {"doc_id": existing.doc_id, "added_chunks": 0, "skipped": True, "reason": "duplicate"}

        index = self._ensure_index()

        embeddings = self._encode(texts)
        start_id = len(self._chunks)
        doc_id = str(len(self._docs))
        stored = [
            StoredChunk(id=str(start_id + i), text=t, doc_id=doc_id, source=source, meta=meta)
            for i, t in enumerate(texts)
            if t.strip()
        ]
        if not stored:
            return {"doc_id": None, "added_chunks": 0, "skipped": True, "reason": "empty"}

        embeddings = embeddings[: len(stored)]
        index.add(embeddings)
        self._chunks.extend(stored)
        self._append_chunks(stored)

        doc = StoredDoc(
            doc_id=doc_id,
            filename=filename,
            sha256=sha256,
            source=source,
            n_chars=n_chars,
            n_chunks=len(stored),
            created_at=created_at,
        )
        self._docs.append(doc)
        if sha256:
            self._doc_hash_to_id[sha256] = doc_id
        self._append_doc(doc)

        faiss.write_index(index, str(self.index_path))
        return {"doc_id": doc_id, "added_chunks": len(stored), "skipped": False}

    def search(self, *, query: str, top_k: int) -> list[StoredChunk]:
        query = query.strip()
        if not query:
            return []
        index = self._ensure_index()
        if index.ntotal == 0:
            return []

        q = self._encode([query])
        scores, ids = index.search(q, top_k)
        out: list[StoredChunk] = []
        for idx in ids[0].tolist():
            if idx < 0 or idx >= len(self._chunks):
                continue
            out.append(self._chunks[int(idx)])
        return out
