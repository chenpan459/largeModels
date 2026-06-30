from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.chat import CustomerServiceChat
from app.config import get_config, project_root
from app.ingest import KnowledgeStore
from app.retriever import Retriever
from app.schemas import AskRequest, AskResponse, HealthResponse, IngestResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="客服知识库", version="1.0.0")
store = KnowledgeStore()
retriever = Retriever()
chat_service = CustomerServiceChat()

static_dir = project_root() / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qdrant_ok, count = store.health()
    llama_ok = store.llama.health()
    status = "ok" if qdrant_ok and llama_ok else "degraded"
    return HealthResponse(
        status=status,
        qdrant=qdrant_ok,
        llama=llama_ok,
        collection=store.collection,
        points_count=count,
    )


@app.post("/api/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    docs_dir = get_config()["paths"]["docs_dir"]
    if not Path(docs_dir).exists():
        raise HTTPException(status_code=400, detail=f"文档目录不存在: {docs_dir}")
    try:
        files, chunks = store.ingest_directory(docs_dir)
    except Exception as e:
        logger.exception("ingest failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return IngestResponse(
        files=files,
        chunks=chunks,
        message=f"已入库 {files} 个文件，共 {chunks} 个片段",
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    try:
        sources = retriever.search(body.question)
        result = chat_service.ask(body.question, sources)
        result.session_id = body.session_id
        return result
    except Exception as e:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


def main() -> None:
    import uvicorn

    cfg = get_config()["server"]
    uvicorn.run("app.main:app", host=cfg["host"], port=cfg["port"], reload=False)


if __name__ == "__main__":
    main()
