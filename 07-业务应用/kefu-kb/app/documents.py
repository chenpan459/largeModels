from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.config import get_config
from app.ingest import SUPPORTED_SUFFIX, load_documents

MAX_UPLOAD_BYTES = 512 * 1024


def docs_dir() -> Path:
    return Path(get_config()["paths"]["docs_dir"])


def list_documents() -> list[dict]:
    root = docs_dir()
    items: list[dict] = []
    for doc_id, title, content in load_documents(root):
        path = _find_path_by_doc_id(root, doc_id)
        if path is None:
            continue
        stat = path.stat()
        items.append(
            {
                "doc_id": doc_id,
                "filename": path.name,
                "relative_path": path.relative_to(root).as_posix(),
                "title": title,
                "size": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "preview": content[:200].replace("\n", " "),
            }
        )
    return sorted(items, key=lambda x: x["relative_path"])


def _find_path_by_doc_id(root: Path, doc_id: str) -> Path | None:
    import hashlib

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIX or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if hashlib.md5(rel.encode()).hexdigest()[:12] == doc_id:
            return path
    return None


def save_upload(filename: str, content: bytes) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIX:
        raise ValueError(f"不支持的文件类型: {suffix}，仅支持 {', '.join(sorted(SUPPORTED_SUFFIX))}")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"文件过大，最大 {MAX_UPLOAD_BYTES // 1024} KB")

    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("无效文件名")

    target = docs_dir() / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def delete_document(relative_path: str) -> None:
    root = docs_dir()
    target = (root / relative_path).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise ValueError("非法路径")
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"文档不存在: {relative_path}")
    if target.suffix.lower() not in SUPPORTED_SUFFIX:
        raise ValueError("不支持的文件类型")
    target.unlink()
