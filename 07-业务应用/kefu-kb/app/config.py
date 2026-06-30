from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        path = _ROOT / "config.yaml"
        with path.open(encoding="utf-8") as f:
            _config = yaml.safe_load(f)
        docs = _config["paths"]["docs_dir"]
        if not Path(docs).is_absolute():
            _config["paths"]["docs_dir"] = str(_ROOT / docs)
    return _config


def project_root() -> Path:
    return _ROOT
