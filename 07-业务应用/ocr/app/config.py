from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        with (_ROOT / "config.yaml").open(encoding="utf-8") as f:
            _config = yaml.safe_load(f)
        for key in ("output_dir", "upload_dir"):
            rel = _config["paths"][key]
            if not Path(rel).is_absolute():
                _config["paths"][key] = str(_ROOT / rel)
    return _config


def project_root() -> Path:
    return _ROOT
