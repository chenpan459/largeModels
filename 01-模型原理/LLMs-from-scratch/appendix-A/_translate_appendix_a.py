#!/usr/bin/env python3
"""Translate appendix-A notebooks and README files to Chinese."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

APPENDIX_DIR = Path(__file__).resolve().parent
BOOK_ROOT = APPENDIX_DIR.parent
MD_TRANS_PATH = APPENDIX_DIR / "_appendix_a_md_trans.json"

HEADER_TABLE = """<table style="width:100%">
<tr>
<td style="vertical-align:middle; text-align:left;">
<font size="2">
<a href="http://mng.bz/orYv">《从零构建大语言模型》（Build a Large Language Model From Scratch）</a> 一书的配套代码，作者 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>
<br>代码仓库：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>
</font>
</td>
<td style="vertical-align:middle; text-align:left;">
<a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>
</td>
</tr>
</table>"""

NOTEBOOKS = [
    "01_main-chapter-code/code-part1.ipynb",
    "01_main-chapter-code/code-part2.ipynb",
    "01_main-chapter-code/exercise-solutions.ipynb",
]

README_PATHS = [
    "README.md",
    "01_main-chapter-code/README.md",
    "02_setup-recommendations/README.md",
]

CODE_COMMENT_TRANSLATIONS: dict[str, str] = {
    "# create a 0D tensor (scalar) from a Python integer": "# 从 Python 整数创建 0D 张量（标量）",
    "# create a 1D tensor (vector) from a Python list": "# 从 Python 列表创建 1D 张量（向量）",
    "# create a 2D tensor from a nested Python list": "# 从嵌套 Python 列表创建 2D 张量",
    "# create a 3D tensor from a nested Python list": "# 从嵌套 Python 列表创建 3D 张量",
    "# create a 3D tensor from NumPy array": "# 从 NumPy 数组创建 3D 张量",
    "            # 1st hidden layer": "            # 第一个隐藏层",
    "              # 1st hidden layer": "              # 第一个隐藏层",
    "            # 2nd hidden layer": "            # 第二个隐藏层",
    "              # 2nd hidden layer": "              # 第二个隐藏层",
    "            # output layer": "            # 输出层",
    "              # output layer": "              # 输出层",
    "          ### LOGGING": "          ### 日志",
    "      # Optional model evaluation": "      # 可选：模型评估",
    '# Note that the book originally used the following line, but the "model =" is redundant': '# 注意：书中最初使用下面这行，但 "model =" 是多余的',
    "  # model = model.to(device) # NEW": "  # model = model.to(device) # 新增",
}

# Exact markdown cell translations (English source -> Chinese)
BASE_MD_TRANS: dict[str, str] = {
    "# Appendix A: Introduction to PyTorch (Part 1)\n": "# 附录 A：PyTorch 入门（第 1 部分）\n",
    "# Appendix A: Introduction to PyTorch (Part 2)\n": "# 附录 A：PyTorch 入门（第 2 部分）\n",
    "## A.1 What is PyTorch\n": "## A.1 什么是 PyTorch\n",
    "## A.2 Understanding tensors\n": "## A.2 理解张量\n",
    "### A.2.1 Scalars, vectors, matrices, and tensors\n": "### A.2.1 标量、向量、矩阵与张量\n",
    "### A.2.2 Tensor data types\n": "### A.2.2 张量数据类型\n",
    "### A.2.3 Common PyTorch tensor operations\n": "### A.2.3 常用 PyTorch 张量运算\n",
    "## A.3 Seeing models as computation graphs\n": "## A.3 将模型视为计算图\n",
    "## A.4 Automatic differentiation made easy\n": "## A.4 轻松理解自动微分\n",
    "## A.5 Implementing multilayer neural networks\n": "## A.5 实现多层神经网络\n",
    "## A.6 Setting up efficient data loaders\n": "## A.6 搭建高效数据加载器\n",
    "## A.7 A typical training loop\n": "## A.7 典型训练循环\n",
    "## A.8 Saving and loading models\n": "## A.8 保存与加载模型\n",
    "See [code-part2.ipynb](code-part2.ipynb)\n": "见 [code-part2.ipynb](code-part2.ipynb) / [code-part2_ch.ipynb](code-part2_ch.ipynb)\n",
    "## A.9 Optimizing training performance with GPUs\n": "## A.9 使用 GPU 优化训练性能\n",
    "### A.9.1 PyTorch computations on GPU devices\n": "### A.9.1 在 GPU 设备上进行 PyTorch 计算\n",
    "### A.9.2 Single-GPU training\n": "### A.9.2 单 GPU 训练\n",
    "### A.9.3 Training with multiple GPUs\n": "### A.9.3 多 GPU 训练\n",
    "See [DDP-script.py](DDP-script.py)\n": "见 [DDP-script.py](DDP-script.py)\n",
    "## Exercise A.1\n": "&nbsp;\n## 练习 A.1\n",
    "## Exercise A.2\n": "&nbsp;\n## 练习 A.2\n",
    "## Exercise A.3\n": "&nbsp;\n## 练习 A.3\n",
    "## Exercise A.4\n": "&nbsp;\n## 练习 A.4\n",
    "The [Python Setup Tips](../../setup/01_optional-python-setup-preferences/README.md) document in this repository contains additional recommendations and tips to set up your Python environment.\n\n\n": (
        "本仓库中的 [Python 环境配置建议（Python Setup Tips）](../../setup/01_optional-python-setup-preferences/README.md) "
        "文档提供了更多配置 Python 环境的建议与技巧。\n\n\n"
    ),
    "The [Installing Libraries Used In This Book document](../../setup/02_installing-python-libraries/README.md) and [directory](../../setup/02_installing-python-libraries/) contains utilities to check whether your environment is set up correctly.\n\n\n": (
        "[本书所用库的安装说明（Installing Libraries Used In This Book）](../../setup/02_installing-python-libraries/README.md) "
        "与 [目录](../../setup/02_installing-python-libraries/) 提供了检查环境是否配置正确的工具。\n\n\n"
    ),
}

README_REPLACEMENTS: list[tuple[str, str]] = [
    ("# Appendix A: Introduction to PyTorch\n", "# 附录 A：PyTorch 入门\n"),
    ("## Main Chapter Code\n", "## 主章节代码\n"),
    (
        "- [01_main-chapter-code](01_main-chapter-code) contains the main chapter code\n",
        "- [01_main-chapter-code](01_main-chapter-code) 包含主章节代码\n",
    ),
    ("## Bonus Materials\n", "## 补充材料\n"),
    (
        "- [02_setup-recommendations](02_setup-recommendations) contains Python installation and setup recommendations.\n",
        "- [02_setup-recommendations](02_setup-recommendations) 包含 Python 安装与环境配置建议\n",
    ),
    ("# Appendix A: Introduction to PyTorch\n\n### Main Chapter Code\n", "# 附录 A：PyTorch 入门\n\n### 主章节代码\n"),
    ("### Main Chapter Code\n", "### 主章节代码\n"),
    ("### Optional Code\n", "### 可选代码\n"),
    (
        "- [code-part1.ipynb](code-part1.ipynb) contains all the section A.1 to A.8 code as it appears in the chapter\n",
        "- [code-part1.ipynb](code-part1.ipynb) / [code-part1_ch.ipynb](code-part1_ch.ipynb) 包含 A.1 至 A.8 节在书中出现的全部代码\n",
    ),
    (
        "- [code-part2.ipynb](code-part2.ipynb) contains all the section A.9 GPU code as it appears in the chapter \n",
        "- [code-part2.ipynb](code-part2.ipynb) / [code-part2_ch.ipynb](code-part2_ch.ipynb) 包含 A.9 节 GPU 相关代码（与书中一致）\n",
    ),
    (
        "- [DDP-script.py](DDP-script.py) contains the script to demonstrate multi-GPU usage (note that Jupyter Notebooks only support single GPUs, so this is a script, not a notebook). You can run it as `python DDP-script.py`. If your machine has more than 2 GPUs, run it as `CUDA_VISIBLE_DEVIVES=0,1 python DDP-script.py`.\n",
        "- [DDP-script.py](DDP-script.py) 演示多 GPU 用法的脚本（Jupyter Notebook 仅支持单 GPU，因此以脚本而非 notebook 形式提供）。"
        " 运行方式：`python DDP-script.py`。若机器 GPU 超过 2 块，可使用 `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py`。\n",
    ),
    (
        "- [exercise-solutions.ipynb](exercise-solutions.ipynb) contains the exercise solutions for this chapter\n",
        "- [exercise-solutions.ipynb](exercise-solutions.ipynb) / [exercise-solutions_ch.ipynb](exercise-solutions_ch.ipynb) 包含本章练习解答\n",
    ),
    (
        "- [DDP-script-torchrun.py](DDP-script-torchrun.py) is an optional version of the `DDP-script.py` script that runs via the PyTorch `torchrun` command instead of spawning and managing multiple processes ourselves via `multiprocessing.spawn`. The `torchrun` command has the advantage of automatically handling distributed initialization, including multi-node coordination, which slightly simplifies the setup process. You can use this script via `torchrun --nproc_per_node=2 DDP-script-torchrun.py`\n",
        "- [DDP-script-torchrun.py](DDP-script-torchrun.py) 是 `DDP-script.py` 的可选版本，通过 PyTorch 的 `torchrun` 命令运行，"
        "而非使用 `multiprocessing.spawn` 自行创建并管理多进程。`torchrun` 会自动处理分布式初始化（含多节点协调），配置更简单。"
        " 用法：`torchrun --nproc_per_node=2 DDP-script-torchrun.py`\n",
    ),
    ("## Python and Environment Setup Recommendations\n", "## Python 与环境配置建议\n"),
    (
        "Please see the [README.md](../../setup/README.md) in the [setup](../../setup) directory for Python installation and setup recommendations.\n",
        "有关 Python 安装与环境配置建议，请参见 [setup](../../setup) 目录下的 [README.md](../../setup/README.md)。\n",
    ),
]

README_CH_FOOTER = """

## 中文文档

| 原文 | 中文版 |
|------|--------|
| [README.md](README.md) | [README_ch.md](README_ch.md) |
| 各子目录 `*.ipynb` | 对应 `*_ch.ipynb` |
"""


def load_prior_reuse_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for ch in ("ch02", "ch03", "ch04", "ch05", "ch06", "ch07"):
        ch_dir = BOOK_ROOT / ch
        if not ch_dir.is_dir():
            continue
        for ch_path in ch_dir.rglob("*_ch.ipynb"):
            en_path = Path(str(ch_path).replace("_ch.ipynb", ".ipynb"))
            if not en_path.is_file():
                continue
            en_nb = json.loads(en_path.read_text(encoding="utf-8"))
            zh_nb = json.loads(ch_path.read_text(encoding="utf-8"))
            for ce, cz in zip(en_nb.get("cells", []), zh_nb.get("cells", [])):
                if ce.get("cell_type") != "markdown" or cz.get("cell_type") != "markdown":
                    continue
                te = "".join(ce.get("source", []))
                tz = "".join(cz.get("source", []))
                if te and te != tz and not is_image_only(te):
                    mapping[te] = tz
    return mapping


def load_md_trans() -> dict[str, str]:
    md = dict(BASE_MD_TRANS)
    md.update(load_prior_reuse_map())
    if MD_TRANS_PATH.is_file():
        md.update(json.loads(MD_TRANS_PATH.read_text(encoding="utf-8")))
    return md


def is_image_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if "<img" in stripped and not re.search(r"(?<![\`>])[A-Za-z]{4,}", re.sub(r"<[^>]+>", " ", stripped)):
        return True
    return bool(re.fullmatch(r"(?:\s|<img[^>]*>\s*)+", stripped, flags=re.DOTALL))


def is_header_table(text: str) -> bool:
    return "Supplementary code for the" in text or "《从零构建大语言模型》" in text


def split_md_code_blocks(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    pattern = re.compile(r"```")
    pos = 0
    in_code = False
    for m in pattern.finditer(text):
        chunk = text[pos : m.start()]
        parts.append(("code" if in_code else "prose", chunk))
        in_code = not in_code
        pos = m.end()
    parts.append(("code" if in_code else "prose", text[pos:]))
    return parts


def translate_prose_chunk(chunk: str, md_trans: dict[str, str]) -> str:
    if chunk in md_trans:
        return md_trans[chunk]
    if is_header_table(chunk):
        return HEADER_TABLE + (chunk[chunk.find("\n") :] if chunk.startswith("<table") else "")
    if chunk.strip() in md_trans:
        return md_trans[chunk.strip()]
    return chunk


def translate_markdown(text: str, md_trans: dict[str, str]) -> str:
    if is_header_table(text):
        return HEADER_TABLE
    if text in md_trans:
        return md_trans[text]
    if is_image_only(text):
        return text
    out: list[str] = []
    for kind, chunk in split_md_code_blocks(text):
        if kind == "code":
            out.append(chunk)
        else:
            out.append(translate_prose_chunk(chunk, md_trans))
    result = "".join(out)
    if result in md_trans:
        return md_trans[result]
    return result


def translate_code_cell(source: str) -> str:
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped in CODE_COMMENT_TRANSLATIONS:
            out.append(CODE_COMMENT_TRANSLATIONS[stripped] + ("\n" if line.endswith("\n") else ""))
        else:
            out.append(line)
    return "".join(out)


def has_untranslated_prose(text: str) -> bool:
    if is_image_only(text) or is_header_table(text):
        return False
    for kind, chunk in split_md_code_blocks(text):
        if kind != "prose":
            continue
        cleaned = chunk
        cleaned = re.sub(r"`[^`]*`", " ", cleaned)
        cleaned = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", cleaned)
        cleaned = re.sub(r"\[[^\]]*\]\([^\)]*\)", " ", cleaned)
        cleaned = re.sub(r"https?://\S+", " ", cleaned)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"&nbsp;", " ", cleaned)
        if re.search(r"[\u4e00-\u9fff]", cleaned) and not re.search(
            r"\b(the|and|with|which|this|that|we|you|our|for|are|was|were|have|has|will|can|should|In|The|Next|Note|Let's|See|contains|Please)\b",
            cleaned,
        ):
            continue
        if re.search(
            r"\b(the|and|with|which|this|that|we|you|our|for|are|was|were|have|has|will|can|should|In|The|Next|Note|Let's|See|contains|Please|Introduction|Exercise|Setup|Recommendations|document|directory|Optional|Main|Chapter|Bonus|Materials)\b",
            cleaned,
        ):
            return True
    return False


def translate_notebook(rel: str, md_trans: dict[str, str]) -> tuple[Path, int]:
    en_path = APPENDIX_DIR / rel
    out_path = en_path.with_name(en_path.stem + "_ch.ipynb")
    nb = json.loads(en_path.read_text(encoding="utf-8"))
    untranslated = 0
    new_cells = []
    for cell in nb.get("cells", []):
        cell = dict(cell)
        if cell.get("cell_type") == "markdown":
            raw = "".join(cell.get("source", []))
            translated = translate_markdown(raw, md_trans)
            if has_untranslated_prose(translated):
                untranslated += 1
            src = translated
            if isinstance(cell.get("source"), list):
                cell["source"] = [src] if not src.endswith("\n") else [src]
            else:
                cell["source"] = src
        elif cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            cell["source"] = translate_code_cell(src)
        new_cells.append(cell)

    if rel.endswith("exercise-solutions.ipynb"):
        title_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# 附录 A 练习解答\n"],
        }
        if new_cells and is_header_table("".join(new_cells[0].get("source", []))):
            new_cells = [new_cells[0], title_cell, *new_cells[1:]]
        else:
            new_cells = [title_cell, *new_cells]

    out_nb = dict(nb)
    out_nb["cells"] = new_cells
    out_path.write_text(json.dumps(out_nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out_path, untranslated


def translate_readme(text: str) -> str:
    out = text
    for old, new in README_REPLACEMENTS:
        out = out.replace(old, new)
    if "## 中文文档" not in out:
        out = out.rstrip() + README_CH_FOOTER
    return out + ("\n" if not out.endswith("\n") else "")


def write_readme_ch_files() -> list[Path]:
    created: list[Path] = []
    for rel in README_PATHS:
        src = (APPENDIX_DIR / rel).read_text(encoding="utf-8")
        dst = APPENDIX_DIR / rel.replace("README.md", "README_ch.md")
        dst.write_text(translate_readme(src), encoding="utf-8")
        created.append(dst)
    return created


def save_md_trans_cache(md_trans: dict[str, str]) -> None:
    extra = {k: v for k, v in md_trans.items() if k not in BASE_MD_TRANS}
    if extra:
        MD_TRANS_PATH.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    md_trans = load_md_trans()
    created: list[Path] = []
    untranslated = 0
    for rel in NOTEBOOKS:
        path, n = translate_notebook(rel, md_trans)
        created.append(path)
        untranslated += n
    created.extend(write_readme_ch_files())
    save_md_trans_cache(load_prior_reuse_map())
    print("Notebooks:", len(NOTEBOOKS))
    print("README_ch:", len(README_PATHS))
    print("Untranslated:", untranslated)
    for p in created:
        print(p.relative_to(APPENDIX_DIR))
    return 1 if untranslated else 0


if __name__ == "__main__":
    sys.exit(main())
