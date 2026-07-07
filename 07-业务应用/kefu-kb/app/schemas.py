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


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    relative_path: str
    title: str
    size: int
    updated_at: str
    preview: str


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentInfo]


class UploadResponse(BaseModel):
    filename: str
    relative_path: str
    message: str


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_n: int | None = Field(default=None, ge=1, le=20)


class SearchResponse(BaseModel):
    question: str
    sources: list[SourceChunk]
