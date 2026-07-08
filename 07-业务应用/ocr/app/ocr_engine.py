"""Wrapper for baidu/Unlimited-OCR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from app.config import get_config, project_root


@dataclass
class OCRResult:
    text: str
    output_dir: Path
    source: str | list[str]


class UnlimitedOCREngine:
    """Document parsing with Hugging Face model baidu/Unlimited-OCR."""

    PRESETS = {
        "gundam": {
            "base_size": 1024,
            "image_size": 640,
            "crop_mode": True,
            "ngram_window": 128,
        },
        "base": {
            "base_size": 1024,
            "image_size": 1024,
            "crop_mode": False,
            "ngram_window": 128,
        },
    }

    def __init__(self) -> None:
        cfg = get_config()
        model_cfg = cfg["model"]
        self.model_name = model_cfg["name"]
        self.device = self._resolve_device(model_cfg.get("device", "auto"))
        self.dtype = self._resolve_dtype(model_cfg.get("dtype", "bfloat16"))
        cache_dir = model_cfg.get("cache_dir")
        self.cache_dir = cache_dir or None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        self.model = self.model.eval().to(self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported dtype: {name}")
        return mapping[name]

    def _output_dir(self, name: str) -> Path:
        base = Path(get_config()["paths"]["output_dir"])
        out = base / name
        out.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def _extract_text(raw: Any, output_dir: Path) -> str:
        if isinstance(raw, str) and raw.strip():
            return raw
        for pattern in ("*.md", "*.txt", "*.json"):
            files = sorted(
                output_dir.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for path in files:
                try:
                    text = path.read_text(encoding="utf-8")
                    if text.strip():
                        return text
                except OSError:
                    continue
        return ""

    def parse_image(
        self,
        image_path: str | Path,
        *,
        mode: str | None = None,
        prompt: str | None = None,
        output_name: str | None = None,
    ) -> OCRResult:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        cfg = get_config()
        single_cfg = cfg["single_image"]
        mode = mode or single_cfg.get("mode", "gundam")
        if mode not in self.PRESETS:
            raise ValueError(f"Unknown mode: {mode}. Use gundam or base.")

        preset = self.PRESETS[mode]
        prompt = prompt or single_cfg["prompt"]
        out_dir = self._output_dir(output_name or image_path.stem)

        raw = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(out_dir),
            base_size=single_cfg.get("base_size", preset["base_size"]),
            image_size=single_cfg.get("image_size", preset["image_size"]),
            crop_mode=single_cfg.get("crop_mode", preset["crop_mode"]),
            max_length=single_cfg.get("max_length", 32768),
            no_repeat_ngram_size=single_cfg.get("no_repeat_ngram_size", 35),
            ngram_window=single_cfg.get("ngram_window", preset["ngram_window"]),
            save_results=cfg["output"].get("save_results", True),
        )
        text = self._extract_text(raw, out_dir)
        return OCRResult(text=text, output_dir=out_dir, source=str(image_path))

    def parse_images(
        self,
        image_paths: list[str | Path],
        *,
        prompt: str | None = None,
        output_name: str = "multi",
    ) -> OCRResult:
        paths = [str(Path(p)) for p in image_paths]
        for p in paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"Image not found: {p}")

        cfg = get_config()
        multi_cfg = cfg["multi_page"]
        prompt = prompt or multi_cfg["prompt"]
        out_dir = self._output_dir(output_name)

        raw = self.model.infer_multi(
            self.tokenizer,
            prompt=prompt,
            image_files=paths,
            output_path=str(out_dir),
            image_size=multi_cfg.get("image_size", 1024),
            max_length=multi_cfg.get("max_length", 32768),
            no_repeat_ngram_size=multi_cfg.get("no_repeat_ngram_size", 35),
            ngram_window=multi_cfg.get("ngram_window", 1024),
            save_results=cfg["output"].get("save_results", True),
        )
        text = self._extract_text(raw, out_dir)
        return OCRResult(text=text, output_dir=out_dir, source=paths)

    def parse_pdf(
        self,
        pdf_path: str | Path,
        *,
        prompt: str | None = None,
        dpi: int | None = None,
        output_name: str | None = None,
    ) -> OCRResult:
        from app.pdf_utils import pdf_to_images

        pdf_path = Path(pdf_path)
        dpi = dpi or get_config()["pdf"].get("dpi", 300)
        image_paths = pdf_to_images(pdf_path, dpi=dpi)
        return self.parse_images(
            image_paths,
            prompt=prompt,
            output_name=output_name or pdf_path.stem,
        )
