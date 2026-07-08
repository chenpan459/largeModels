"""PDF to image conversion."""

from __future__ import annotations

import tempfile
from pathlib import Path


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> list[str]:
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_pdf_"))
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    paths: list[str] = []

    try:
        for i, page in enumerate(doc):
            out = tmp_dir / f"page_{i + 1:04d}.png"
            page.get_pixmap(matrix=matrix).save(str(out))
            paths.append(str(out))
    finally:
        doc.close()

    return paths
