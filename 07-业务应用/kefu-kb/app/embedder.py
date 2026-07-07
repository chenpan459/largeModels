from __future__ import annotations

import logging
from typing import Protocol

from app.config import get_config
from app.llama_client import LlamaClient

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def health(self) -> bool: ...


class LlamaEmbedder:
    def __init__(self) -> None:
        self._client = LlamaClient()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(texts)

    def health(self) -> bool:
        return self._client.health()


class LocalEmbedder:
    """本地 SentenceTransformer，无需 llama-server 即可入库/检索。"""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def health(self) -> bool:
        return True


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        cfg = get_config().get("embedding", {})
        backend = cfg.get("backend", "llama")
        if backend == "local":
            model_name = cfg.get("local_model", "paraphrase-multilingual-MiniLM-L12-v2")
            _embedder = LocalEmbedder(model_name)
        else:
            _embedder = LlamaEmbedder()
    return _embedder
