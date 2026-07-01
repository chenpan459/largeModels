#!/usr/bin/env python3
"""Translate ch05 markdown content (README and notebook cells) from English to Chinese."""

import json
import re
import sys
import time
from pathlib import Path

CH05 = Path(__file__).resolve().parent

# Cells/patterns to skip (already translated or should stay English)
SKIP_PATTERNS = [
    r"^<table",
    r"^<img ",
    r"^!\[",
    r"^```",
    r"^http",
    r"^https",
    r"^Ep \d",
    r"^ubuntu@",
    r"^Skipping ",
    r"^100%\|",
    r"^42 file",
    r"^Total ",
    r"^Tokenizing ",
    r"^Training",
    r"^Saved ",
    r"^Book processed",
    r"^Every effort",
    r"^Allocated memory",
    r"^Reserved memory",
    r"^PyTorch version",
    r"^Using cuda",
    r"^CUDA version",
]

# Preserve these tokens during translation
PRESERVE = [
    "GPT", "LLM", "LLMs", "PyTorch", "TensorFlow", "OpenAI", "Hugging Face",
    "Chainlit", "Llama", "Qwen", "Gemma", "Olmo", "Muon", "GaLore", "AdamW",
    "DDP", "KV", "MoE", "RoPE", "NoPE", "RMSNorm", "LayerNorm", "SwiGLU",
    "BPE", "tiktoken", "safetensors", "transformers", "CUDA", "GPU", "VRAM",
    "CPU", "WSL", "JSON", "Python", "Jupyter", "notebook", "pip", "bash",
    "GitHub", "HuggingFace", "Meta AI", "Cohere", "Tiny Aya", "Gutenberg",
    "Chainlit", "Weights and Biases", "Flash Attention", "torch.compile",
    "thunder", "Windows", "Linux", "macOS", "Ubuntu", "Docker",
]


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def chinese_ratio(text: str) -> float:
    if not text.strip():
        return 1.0
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    total = cn + en
    return cn / total if total else 1.0


def should_skip(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    for pat in SKIP_PATTERNS:
        if re.match(pat, stripped, re.IGNORECASE):
            return True
    # Mostly code/output
    if stripped.count("`") >= 4 and len(stripped) < 200:
        return True
    return False


def protect_terms(text: str) -> tuple[str, dict]:
    mapping = {}
    result = text
    for i, term in enumerate(PRESERVE):
        placeholder = f"__KEEP_{i}__"
        if term in result:
            mapping[placeholder] = term
            result = result.replace(term, placeholder)
    return result, mapping


def restore_terms(text: str, mapping: dict) -> str:
    for placeholder, term in mapping.items():
        text = text.replace(placeholder, term)
    return text


def translate_text(text: str, translator) -> str:
    if should_skip(text):
        return text
    if chinese_ratio(text) > 0.5:
        return text

    protected, mapping = protect_terms(text)
    # Split long text into chunks (Google Translate limit ~5000 chars)
    chunks = []
    lines = protected.split("\n")
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > 4500:
            if current:
                chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)

    translated_parts = []
    for chunk in chunks:
        if not chunk.strip():
            translated_parts.append(chunk)
            continue
        try:
            result = translator.translate(chunk)
            translated_parts.append(result)
            time.sleep(0.1)
        except Exception as e:
            print(f"  Translation error: {e}", file=sys.stderr)
            translated_parts.append(chunk)

    result = "\n".join(translated_parts)
    return restore_terms(result, mapping)


def translate_readme(en_path: Path, ch_path: Path, translator) -> bool:
    if not en_path.exists():
        return False
    en_text = en_path.read_text(encoding="utf-8")
    if ch_path.exists():
        ch_text = ch_path.read_text(encoding="utf-8")
        if chinese_ratio(ch_text) > 0.6 and "## 中文文档" not in ch_text:
            print(f"  Skip README (already translated): {ch_path.relative_to(CH05)}")
            return False

    print(f"  Translating README: {en_path.relative_to(CH05)}")
    translated = translate_text(en_text, translator)
    ch_path.write_text(translated, encoding="utf-8")
    return True


def translate_notebook(en_path: Path, ch_path: Path, translator) -> bool:
    if not en_path.exists():
        return False

    with open(en_path, encoding="utf-8") as f:
        en_nb = json.load(f)
    if ch_path.exists():
        with open(ch_path, encoding="utf-8") as f:
            ch_nb = json.load(f)
    else:
        ch_nb = json.loads(json.dumps(en_nb))

    changed = False
    en_cells = en_nb.get("cells", [])
    ch_cells = ch_nb.get("cells", [])

    if len(en_cells) != len(ch_cells):
        ch_cells = json.loads(json.dumps(en_cells))
        ch_nb["cells"] = ch_cells

    for i, (en_cell, ch_cell) in enumerate(zip(en_cells, ch_cells)):
        if en_cell.get("cell_type") != "markdown":
            continue
        en_source = "".join(en_cell.get("source", []))
        ch_source = "".join(ch_cell.get("source", []))

        # Keep already-good Chinese cells; retranslate low-quality ones
        if chinese_ratio(ch_source) > 0.55 and has_chinese(ch_source):
            continue

        if should_skip(en_source):
            if ch_source != en_source:
                ch_cell["source"] = en_cell["source"]
                changed = True
            continue

        print(f"    Cell {i}: translating...")
        new_source = translate_text(en_source, translator)
        if new_source != ch_source:
            # Preserve notebook source format (list of lines)
            if isinstance(en_cell["source"], list):
                lines = new_source.splitlines(keepends=True)
                if lines and not lines[-1].endswith("\n"):
                    pass
                ch_cell["source"] = lines if lines else [new_source]
            else:
                ch_cell["source"] = new_source
            changed = True
        time.sleep(0.05)

    if changed:
        with open(ch_path, "w", encoding="utf-8") as f:
            json.dump(ch_nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"  Updated notebook: {ch_path.relative_to(CH05)}")
    else:
        print(f"  No changes: {ch_path.relative_to(CH05)}")
    return changed


def main():
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("Install: pip install deep-translator", file=sys.stderr)
        sys.exit(1)

    translator = GoogleTranslator(source="en", target="zh-CN")

    print("=== Translating README files ===")
    for readme in sorted(CH05.rglob("README.md")):
        ch_readme = readme.parent / "README_ch.md"
        translate_readme(readme, ch_readme, translator)

    print("\n=== Translating notebooks ===")
    for ipynb in sorted(CH05.rglob("*.ipynb")):
        if ipynb.name.endswith("_ch.ipynb"):
            continue
        ch_ipynb = ipynb.with_name(ipynb.stem + "_ch.ipynb")
        print(f"  Notebook: {ipynb.relative_to(CH05)}")
        translate_notebook(ipynb, ch_ipynb, translator)

    print("\nDone!")


if __name__ == "__main__":
    main()
