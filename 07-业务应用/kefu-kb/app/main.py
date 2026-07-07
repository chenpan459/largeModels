from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.chat import CustomerServiceChat
from app.config import get_config, project_root
from app.documents import delete_document, list_documents, save_upload
from app.ingest import KnowledgeStore
from app.retriever import Retriever
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="客服知识库", version="1.1.0")
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
    embed_ok = store.embedder.health()
    llama_ok = store.llama.health()
    status = "ok" if qdrant_ok and embed_ok else "degraded"
    return HealthResponse(
        status=status,
        qdrant=qdrant_ok,
        llama=llama_ok,
        collection=store.collection,
        points_count=count,
    )


@app.get("/api/docs", response_model=DocumentListResponse)
def list_docs() -> DocumentListResponse:
    docs = [DocumentInfo(**item) for item in list_documents()]
    return DocumentListResponse(total=len(docs), documents=docs)


@app.post("/api/docs/upload", response_model=UploadResponse)
async def upload_doc(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    try:
        content = await file.read()
        path = save_upload(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    rel = path.relative_to(Path(get_config()["paths"]["docs_dir"])).as_posix()
    return UploadResponse(
        filename=path.name,
        relative_path=rel,
        message=f"已上传 {path.name}，请点击「重新入库」使文档生效",
    )


@app.delete("/api/docs/{relative_path:path}")
def remove_doc(relative_path: str) -> dict[str, str]:
    try:
        delete_document(relative_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"message": f"已删除 {relative_path}，请重新入库以更新向量索引"}


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


@app.post("/api/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    try:
        top_n = body.top_n if body.top_n is not None else retriever.top_n
        sources = retriever.search(body.question)[:top_n]
        return SearchResponse(question=body.question, sources=sources)
    except Exception as e:
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


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
