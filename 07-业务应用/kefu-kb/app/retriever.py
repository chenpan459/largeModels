from __future__ import annotations

from qdrant_client import QdrantClient

from app.config import get_config
from app.llama_client import LlamaClient
from app.schemas import SourceChunk


class Retriever:
    def __init__(self) -> None:
        cfg = get_config()
        q = cfg["qdrant"]
        self.collection = q["collection"]
        self.client = QdrantClient(host=q["host"], port=q["port"])
        self.llama = LlamaClient()
        self.top_k = cfg["rag"]["top_k"]
        self.top_n = cfg["rag"]["top_n"]
        self.use_rerank = cfg["rag"].get("use_rerank", True)
        self.score_threshold = cfg["rag"].get("score_threshold", 0.35)

    def search(self, question: str) -> list[SourceChunk]:
        query_vector = self.llama.embed([question])[0]
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=self.top_k,
        )

        if not hits:
            return []

        candidates = [
            SourceChunk(
                doc_id=str(h.payload.get("doc_id", "")),
                title=str(h.payload.get("title", "")),
                text=str(h.payload.get("text", "")),
                score=float(h.score or 0),
            )
            for h in hits
        ]

        if self.use_rerank and len(candidates) > 1:
            try:
                docs = [c.text for c in candidates]
                results = self.llama.rerank(question, docs, self.top_n)
                reranked: list[SourceChunk] = []
                for item in results:
                    idx = item.get("index", 0)
                    if 0 <= idx < len(candidates):
                        c = candidates[idx].model_copy()
                        c.score = float(item.get("relevance_score", c.score))
                        reranked.append(c)
                candidates = reranked or candidates[: self.top_n]
            except Exception:
                candidates = candidates[: self.top_n]
        else:
            candidates = candidates[: self.top_n]

        return candidates[: self.top_n]
