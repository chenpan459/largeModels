#!/usr/bin/env python3
"""Offline markdown translation for ch06 *_ch.ipynb files."""

import json
import re
import sys
from pathlib import Path

CH06 = Path(__file__).resolve().parent

HEADER_EN = (
    'Supplementary code for the <a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> book by <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n'
    '<br>Code repository: <a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>'
)
HEADER_CH = (
    '<a href="http://mng.bz/orYv">《从零构建大语言模型》（Build a Large Language Model From Scratch）</a> 一书的配套代码，作者 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n'
    '<br>代码仓库：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>'
)

SECTION_HEADINGS = {
    "# Chapter 6: Finetuning for Classification": "# 第 6 章：分类微调",
    "# Load And Use Finetuned Model": "# 加载并使用微调后的模型",
    "# Scikit-learn Logistic Regression Model": "# Scikit-learn 逻辑回归模型",
    "## Scikit-learn baseline": "## Scikit-learn 基线",
    "### 6.1 Different categories of finetuning": "### 6.1 微调的不同类别",
    "### 6.2 Preparing the dataset": "### 6.2 准备数据集",
    "### 6.3 Creating data loaders": "### 6.3 创建数据加载器",
    "### 6.4 Initializing a model with pretrained weights": "### 6.4 用预训练权重初始化模型",
    "### 6.5 Adding a classification head": "### 6.5 添加分类头",
    "### 6.6 Calculating the classification loss and accuracy": "### 6.6 计算分类损失与准确率",
    "### 6.7 Finetuning the model on supervised data": "### 6.7 在有标注数据上微调模型",
    "### 6.8 Using the LLM as a spam classifier": "### 6.8 将 LLM 用作垃圾短信分类器",
    "# Exercise solutions for Chapter 6": "# 第 6 章练习解答",
    "## Exercise 6.1: Increase the context length": "## 练习 6.1：增加上下文长度",
    "## Exercise 6.2: Finetune the whole model": "## 练习 6.2：微调整个模型",
    "## Exercise 6.3: Finetune the first versus last output token": "## 练习 6.3：微调第一个 token 与最后一个 token",
}

PHRASES = [
    ("This notebook contains minimal code to load the finetuned model that was created and saved in chapter 6 via",
     "本 notebook 含最少代码，用于加载第 6 章通过"),
    ("There is no code in this section", "本节无代码"),
    ("Note that", "注意，"),
    ("However,", "不过，"),
    ("In this section", "在本节中"),
    ("This section", "本节"),
    ("finetuning", "微调"),
    ("Finetuning", "微调"),
    ("classification", "分类"),
    ("Classification", "分类"),
    ("pretrained weights", "预训练权重"),
    ("spam", "垃圾短信"),
    ("dataset", "数据集"),
    ("tokenizer", "分词器"),
    ("training", "训练"),
    ("validation", "验证"),
    ("accuracy", "准确率"),
]


def chinese_ratio(text: str) -> float:
    if not text.strip():
        return 1.0
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    total = cn + en
    return cn / total if total else 1.0


def translate_markdown(text: str, en_text: str | None = None) -> str:
    if chinese_ratio(text) >= 0.55:
        return text
    result = text.replace(HEADER_EN, HEADER_CH)
    for en, ch in SECTION_HEADINGS.items():
        result = result.replace(en, ch)
    for en, ch in PHRASES:
        result = result.replace(en, ch)
    if chinese_ratio(result) < 0.35 and en_text:
        result = en_text.replace(HEADER_EN, HEADER_CH)
        for en, ch in SECTION_HEADINGS.items():
            result = result.replace(en, ch)
        for en, ch in PHRASES:
            result = result.replace(en, ch)
    return result


def process_notebook(en_path: Path, ch_path: Path) -> int:
    en_nb = json.loads(en_path.read_text(encoding="utf-8"))
    ch_nb = json.loads(ch_path.read_text(encoding="utf-8")) if ch_path.exists() else json.loads(json.dumps(en_nb))
    en_cells, ch_cells = en_nb["cells"], ch_nb["cells"]
    if len(en_cells) != len(ch_cells):
        ch_cells = json.loads(json.dumps(en_cells))
        ch_nb["cells"] = ch_cells
    changes = 0
    for en_cell, ch_cell in zip(en_cells, ch_cells):
        if en_cell.get("cell_type") != "markdown":
            continue
        en_src = "".join(en_cell.get("source", []))
        ch_src = "".join(ch_cell.get("source", []))
        new_src = translate_markdown(ch_src, en_src)
        if new_src != ch_src:
            lines = new_src.splitlines(keepends=True)
            ch_cell["source"] = lines if lines else [new_src]
            changes += 1
    if changes:
        ch_path.write_text(json.dumps(ch_nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changes


def main():
    total = 0
    for ipynb in sorted(CH06.rglob("*.ipynb")):
        if ipynb.name.endswith("_ch.ipynb"):
            continue
        ch_ipynb = ipynb.with_name(ipynb.stem + "_ch.ipynb")
        if not ch_ipynb.exists():
            continue
        n = process_notebook(ipynb, ch_ipynb)
        if n:
            print(f"Updated {ch_ipynb.relative_to(CH06)}: {n} cells")
            total += n
    print(f"Done: {total} cells updated")


if __name__ == "__main__":
    main()
