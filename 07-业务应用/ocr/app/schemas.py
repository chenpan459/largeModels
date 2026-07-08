from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    cuda_available: bool


class OCRResponse(BaseModel):
    text: str
    output_dir: str
    source: str | list[str] = Field(description="Input file path(s)")


class OCRImageRequest(BaseModel):
    mode: str = "gundam"
    prompt: str | None = None
