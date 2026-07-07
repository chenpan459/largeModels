from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from app.config import get_config, project_root


def create_qdrant_client() -> QdrantClient:
    q = get_config()["qdrant"]
    local_path = q.get("local_path")
    if local_path:
        p = Path(local_path) if Path(str(local_path)).is_absolute() else project_root() / local_path
        p.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(p), check_compatibility=False)
    return QdrantClient(host=q["host"], port=q["port"], check_compatibility=False)
