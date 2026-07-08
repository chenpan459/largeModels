"""FastAPI OCR service."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_config, project_root
from app.ocr_engine import UnlimitedOCREngine
from app.schemas import HealthResponse, OCRResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Unlimited OCR", version="1.0.0")
_engine: UnlimitedOCREngine | None = None

static_dir = project_root() / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def get_engine() -> UnlimitedOCREngine:
    global _engine
    if _engine is None:
        logger.info("Loading baidu/Unlimited-OCR...")
        _engine = UnlimitedOCREngine()
    return _engine


def _save_upload(file: UploadFile, suffix: str) -> Path:
    upload_dir = Path(get_config()["paths"]["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}{suffix}"
    path = upload_dir / name
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg = get_config()
    cuda = torch.cuda.is_available()
    device = cfg["model"].get("device", "auto")
    if device == "auto":
        device = "cuda" if cuda else "cpu"
    return HealthResponse(
        status="ok",
        model=cfg["model"]["name"],
        device=device,
        cuda_available=cuda,
    )


@app.post("/api/ocr/image", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    mode: str = Form("gundam"),
    prompt: str | None = Form(None),
) -> OCRResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    suffix = Path(file.filename).suffix or ".png"
    path = _save_upload(file, suffix)
    try:
        result = get_engine().parse_image(path, mode=mode, prompt=prompt or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        path.unlink(missing_ok=True)
    return OCRResponse(
        text=result.text,
        output_dir=str(result.output_dir),
        source=result.source,
    )


@app.post("/api/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf(
    file: UploadFile = File(...),
    prompt: str | None = Form(None),
) -> OCRResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    path = _save_upload(file, ".pdf")
    try:
        result = get_engine().parse_pdf(path, prompt=prompt or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        path.unlink(missing_ok=True)
    return OCRResponse(
        text=result.text,
        output_dir=str(result.output_dir),
        source=result.source,
    )


@app.post("/api/ocr/images", response_model=OCRResponse)
async def ocr_images(
    files: list[UploadFile] = File(...),
    prompt: str | None = Form(None),
) -> OCRResponse:
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一张图片")
    paths: list[Path] = []
    try:
        for f in files:
            suffix = Path(f.filename or "page.png").suffix or ".png"
            paths.append(_save_upload(f, suffix))
        result = get_engine().parse_images(
            [str(p) for p in paths],
            prompt=prompt or None,
            output_name=uuid.uuid4().hex[:8],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        for p in paths:
            p.unlink(missing_ok=True)
    return OCRResponse(
        text=result.text,
        output_dir=str(result.output_dir),
        source=result.source,
    )


if __name__ == "__main__":
    import uvicorn

    cfg = get_config()["server"]
    uvicorn.run(
        "app.main:app",
        host=cfg.get("host", "0.0.0.0"),
        port=cfg.get("port", 8010),
        reload=False,
    )
