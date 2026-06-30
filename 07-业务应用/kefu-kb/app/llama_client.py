from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_config

logger = logging.getLogger(__name__)


class LlamaClient:
    def __init__(self) -> None:
        cfg = get_config()["llama"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.api_key = cfg.get("api_key", "no-key")
        self.chat_model = cfg.get("chat_model", "default")
        self.embed_model = cfg.get("embed_model", "default")
        self.rerank_model = cfg.get("rerank_model", "default")
        self.timeout = cfg.get("timeout", 120)
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def health(self) -> bool:
        try:
            with self._client() as c:
                r = c.get("/health")
                return r.status_code == 200
        except Exception as e:
            logger.warning("llama-server health check failed: %s", e)
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._client() as c:
            r = c.post(
                "/v1/embeddings",
                headers=self._headers,
                json={"input": texts, "model": self.embed_model},
            )
            r.raise_for_status()
            data = r.json()["data"]
            data.sort(key=lambda x: x["index"])
            return [item["embedding"] for item in data]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict[str, Any]]:
        if not documents:
            return []
        with self._client() as c:
            r = c.post(
                "/v1/rerank",
                headers=self._headers,
                json={
                    "model": self.rerank_model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )
            r.raise_for_status()
            return r.json().get("results", [])

    def chat(self, system: str, user: str) -> str:
        with self._client() as c:
            r = c.post(
                "/v1/chat/completions",
                headers=self._headers,
                json={
                    "model": self.chat_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
