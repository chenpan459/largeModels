from __future__ import annotations

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float = 0.0


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    session_id: str | None = None


class IngestResponse(BaseModel):
    files: int
    chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    llama: bool
    collection: str
    points_count: int = 0
