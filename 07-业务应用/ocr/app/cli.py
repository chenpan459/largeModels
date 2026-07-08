"""Command-line interface for OCR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Document OCR with baidu/Unlimited-OCR",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    img = sub.add_parser("image", help="Parse a single image")
    img.add_argument("path", help="Image file path")
    img.add_argument(
        "--mode",
        choices=["gundam", "base"],
        default=None,
        help="gundam: crop mode; base: full image",
    )
    img.add_argument("--prompt", default=None)

    pdf = sub.add_parser("pdf", help="Parse a PDF document")
    pdf.add_argument("path", help="PDF file path")
    pdf.add_argument("--dpi", type=int, default=None)
    pdf.add_argument("--prompt", default=None)

    multi = sub.add_parser("images", help="Parse multiple page images")
    multi.add_argument("paths", nargs="+", help="Image file paths")
    multi.add_argument("--prompt", default=None)

    args = parser.parse_args()

    from app.ocr_engine import UnlimitedOCREngine

    print("Loading model (first run downloads ~6GB)...", file=sys.stderr)
    engine = UnlimitedOCREngine()

    if args.command == "image":
        result = engine.parse_image(
            args.path,
            mode=args.mode,
            prompt=args.prompt,
        )
    elif args.command == "pdf":
        result = engine.parse_pdf(
            args.path,
            dpi=args.dpi,
            prompt=args.prompt,
        )
    else:
        result = engine.parse_images(args.paths, prompt=args.prompt)

    print(f"# source: {result.source}", file=sys.stderr)
    print(f"# output: {result.output_dir}", file=sys.stderr)
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
