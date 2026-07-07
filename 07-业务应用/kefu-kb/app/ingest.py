from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import get_config
from app.embedder import get_embedder
from app.llama_client import LlamaClient
from app.qdrant_client import create_qdrant_client

SUPPORTED_SUFFIX = {".md", ".txt", ".markdown"}


@dataclass
class DocChunk:
    doc_id: str
    title: str
    text: str
    chunk_index: int


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at > start + chunk_size // 2:
                end = split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_documents(docs_dir: str | Path) -> list[tuple[str, str, str]]:
    root = Path(docs_dir)
    if not root.exists():
        return []

    docs: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIX or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        doc_id = hashlib.md5(rel.encode()).hexdigest()[:12]
        title = path.stem
        content = path.read_text(encoding="utf-8")
        first_line = content.split("\n", 1)[0].strip()
        if first_line.startswith("#"):
            title = first_line.lstrip("#").strip()
        docs.append((doc_id, title, content))
    return docs


def build_chunks(docs_dir: str | Path) -> list[DocChunk]:
    cfg = get_config()["rag"]
    chunk_size = cfg["chunk_size"]
    overlap = cfg["chunk_overlap"]
    chunks: list[DocChunk] = []

    for doc_id, title, content in load_documents(docs_dir):
        for i, piece in enumerate(split_text(content, chunk_size, overlap)):
            chunks.append(DocChunk(doc_id=doc_id, title=title, text=piece, chunk_index=i))
    return chunks


class KnowledgeStore:
    def __init__(self) -> None:
        cfg = get_config()
        q = cfg["qdrant"]
        self.collection = q["collection"]
        self.client = create_qdrant_client()
        self.llama = LlamaClient()
        self.embedder = get_embedder()
        self.vector_size: int | None = None

    def health(self) -> tuple[bool, int]:
        try:
            info = self.client.get_collection(self.collection)
            return True, info.points_count
        except Exception:
            return False, 0

    def ensure_collection(self, vector_size: int) -> None:
        self.vector_size = vector_size
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection in names:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def ingest_directory(self, docs_dir: str | Path) -> tuple[int, int]:
        chunks = build_chunks(docs_dir)
        if not chunks:
            return 0, 0

        texts = [c.text for c in chunks]
        batch_size = 32
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            all_vectors.extend(self.embedder.embed(texts[i : i + batch_size]))

        vector_size = len(all_vectors[0])
        self.ensure_collection(vector_size)

        points = []
        for chunk, vector in zip(chunks, all_vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.doc_id}:{chunk.chunk_index}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection, points=points)
        files = len(load_documents(docs_dir))
        return files, len(chunks)
