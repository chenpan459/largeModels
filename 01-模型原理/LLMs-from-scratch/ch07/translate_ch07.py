#!/usr/bin/env python3
"""Offline markdown translation for ch07 *_ch.ipynb files."""

import json
import re
import sys
import time
from pathlib import Path

CH07 = Path(__file__).resolve().parent

HEADER_EN = (
    'Supplementary code for the <a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> book by <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n'
    '<br>Code repository: <a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>'
)
HEADER_CH = (
    '<a href="http://mng.bz/orYv">《从零构建大语言模型》（Build a Large Language Model From Scratch）</a> 一书的配套代码，作者 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n'
    '<br>代码仓库：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>'
)

SECTION_HEADINGS = {
    "# Chapter 7: Finetuning To Follow Instructions": "# 第 7 章：微调以遵循指令",
    "# Load And Use Finetuned Model": "# 加载并使用微调后的模型",
    "# Exercise solutions for Chapter 7": "# 第 7 章练习解答",
    "## 7.1 Introduction to instruction finetuning": "## 7.1 指令微调简介",
    "## 7.2 Preparing a dataset for supervised instruction finetuning": "## 7.2 准备用于监督式指令微调的数据集",
    "## 7.3 Organizing data into training batches": "## 7.3 将数据组织为训练 batch",
    "## 7.4 Creating data loaders for an instruction dataset": "## 7.4 为指令数据集创建数据加载器",
    "## 7.5 Loading a pretrained LLM": "## 7.5 加载预训练 LLM",
    "## 7.6 Finetuning the LLM on instruction data": "## 7.6 在指令数据上微调 LLM",
    "## 7.7 Extracting and saving responses": "## 7.7 提取并保存回复",
    "## 7.8 Evaluating the finetuned LLM": "## 7.8 评估微调后的 LLM",
    "## 7.9 Conclusions": "## 7.9 结论",
    "### 7.9.1 What's next": "### 7.9.1 接下来做什么",
    "### 7.9.2 Staying up to date in a fast-moving field": "### 7.9.2 在快速变化的领域中保持更新",
    "### 7.9.3 Final words": "### 7.9.3 结语",
    "## Summary and takeaways": "## 总结与要点",
    "## What's next?": "## 接下来做什么？",
    "## Exercise 7.1: Change the prompt style": "## 练习 7.1：更换提示风格",
    "#### Prompt: Consider special tokens": "#### 提示：考虑特殊 token",
    "## Exercise 7.2: Finetune on the original Alpaca dataset": "## 练习 7.2：在原始 Alpaca 数据集上微调",
    "## Exercise 7.3: Finetune on a different instruction dataset": "## 练习 7.3：在不同指令数据集上微调",
    "## Exercise 7.4: Parameter-efficient finetuning with LoRA": "## 练习 7.4：使用 LoRA 的参数高效微调",
    "# Generating An Instruction Dataset via Llama 3 and Ollama": "# 通过 Llama 3 与 Ollama 生成指令数据集",
    "## Installing Ollama and Downloading Llama 3": "## 安装 Ollama 并下载 Llama 3",
    "## Installing Ollama and Downloading Llama 3.1": "## 安装 Ollama 并下载 Llama 3.1",
    "## Using Ollama's REST API": "## 使用 Ollama 的 REST API",
    "## Extract Instructions": "## 提取指令",
    "## Generate Responses": "## 生成回复",
    "## Generate Dataset": "## 生成数据集",
    "# Improving Instruction-Data Via Reflection-Tuning Using GPT-4": "# 使用 GPT-4 通过反思微调改进指令数据",
    "## Test OpenAI API": "## 测试 OpenAI API",
    "## Load JSON Entries": "## 加载 JSON 条目",
    "## Improve Instructions": "## 改进指令",
    "## Improve Responses": "## 改进回复",
    "## Improving the Dataset": "## 改进数据集",
    "### Reflect instructions": "### 反思指令",
    "### Reflect responses": "### 反思回复",
    "## Creating Improved Instruction Data": "## 创建改进的指令数据",
    "# Evaluating Instruction Responses Using OpenAI API": "# 使用 OpenAI API 评估指令回复",
    "# Evaluating Instruction Responses Locally Using Llama 3 Models via Ollama": "# 通过 Ollama 在本地用 Llama 3 模型评估指令回复",
    "# Correlation Analysis of Scores": "# 评分相关性分析",
    "## GPT-4 vs Llama 3 8B": "## GPT-4 与 Llama 3 8B 对比",
    "### Correlation coefficients": "### 相关系数",
    "# Creating \"Passive Voice\" Entries for an Instruction Dataset": "# 为指令数据集创建「被动语态」条目",
    "# Generating A Preference Dataset With Llama 3.1 70B And Ollama": "# 使用 Llama 3.1 70B 与 Ollama 生成偏好数据集",
    "# Direct Preference Optimization (DPO) for LLM Alignment (From Scratch)": "# 从零实现用于 LLM 对齐的直接偏好优化（DPO）",
    "# 1) A brief introduction to DPO": "# 1) DPO 简介",
    "# 2) Preparing a preference dataset for DPO": "# 2) 为 DPO 准备偏好数据集",
    "## 2.1) Loading a preference dataset": "## 2.1) 加载偏好数据集",
    "## 2.2) Creating training, validation, and test splits": "## 2.2) 创建训练、验证与测试划分",
    "## 2.3) Developing a `PreferenceDataset` class and batch processing function": "## 2.3) 实现 `PreferenceDataset` 类与 batch 处理函数",
    "## 2.4 Creating training, validation, and test set data loaders": "## 2.4 创建训练、验证与测试集数据加载器",
    "# 3) Loading a finetuned LLM for DPO alignment": "# 3) 加载用于 DPO 对齐的微调 LLM",
    "# 4) Coding the DPO Loss Function": "# 4) 实现 DPO 损失函数",
    "# 5) Training the model": "# 5) 训练模型",
    "# 6) Analyzing the results": "# 6) 分析结果",
}

SKIP_PATTERNS = [
    r"^<table",
    r"^<img ",
    r"^!\[",
    r"^&nbsp;\s*$",
    r"^---\s*$",
]

PRESERVE = [
    "GPT", "LLM", "LLMs", "PyTorch", "TensorFlow", "OpenAI", "Ollama", "Llama",
    "Alpaca", "Phi-3", "Magpie", "DPO", "LoRA", "JSON", "Python", "Jupyter",
    "notebook", "REST", "API", "GitHub", "Meta AI", "CUDA", "GPU", "CPU",
    "macOS", "Windows", "Linux", "tiktoken", "SpamDataset", "InstructionDataset",
    "PreferenceDataset", "GPTModel", "Llama3", "Chainlit", "llama.cpp",
    "GPT-2", "GPT-4", "M3", "MacBook", "Hugging Face", "Safetensors",
]

MIXED_EN_WORDS = re.compile(
    r"\b(?:the|we|this|that|with|for|and|is|are|was|were|will|can|not|"
    r"In|Note|However|Similar|First|Next|Lastly|Above|Below|Using|The|It|"
    r"They|Also|Here|When|If|As|This|These|Those|Before|After|During|While|"
    r"Since|Because|Although|Through|From|Into|About|Between|Under|Over|"
    r"Within|Similar|Concretely|Alternatively|Prior|Linux|macOS|Windows)\b"
)


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def chinese_ratio(text: str) -> float:
    if not text.strip():
        return 1.0
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    total = cn + en
    return cn / total if total else 1.0


def is_mixed(text: str) -> bool:
    if not has_chinese(text):
        return False
    return len(MIXED_EN_WORDS.findall(text)) >= 3


def is_fully_translated(text: str) -> bool:
    if should_skip(text):
        return True
    if not has_chinese(text):
        return False
    if is_mixed(text):
        return False
    return chinese_ratio(text) >= 0.70


def should_skip(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    for pat in SKIP_PATTERNS:
        if re.match(pat, stripped, re.IGNORECASE):
            return True
    if stripped.startswith("<img") and stripped.endswith(">"):
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


def apply_headings(text: str) -> str:
    result = text.replace(HEADER_EN, HEADER_CH)
    for en, ch in SECTION_HEADINGS.items():
        result = result.replace(en, ch)
    return result


def google_translate(text: str, translator) -> str:
    protected, mapping = protect_terms(text)
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
            translated_parts.append(translator.translate(chunk))
            time.sleep(0.08)
        except Exception as exc:
            print(f"  Translation error: {exc}", file=sys.stderr)
            translated_parts.append(chunk)
    return restore_terms("\n".join(translated_parts), mapping)


def translate_text(en_text: str, translator) -> str:
    if should_skip(en_text):
        return apply_headings(en_text)
    if translator is None:
        return apply_headings(en_text)
    translated = google_translate(en_text, translator)
    return apply_headings(translated)


def source_to_list(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]


def process_notebook(en_path: Path, ch_path: Path, translator=None) -> int:
    en_nb = json.loads(en_path.read_text(encoding="utf-8"))
    ch_nb = json.loads(ch_path.read_text(encoding="utf-8")) if ch_path.exists() else json.loads(json.dumps(en_nb))
    en_cells, ch_cells = en_nb["cells"], ch_nb["cells"]
    if len(en_cells) != len(ch_cells):
        ch_cells = json.loads(json.dumps(en_cells))
        ch_nb["cells"] = ch_cells
    changes = 0
    for idx, (en_cell, ch_cell) in enumerate(zip(en_cells, ch_cells)):
        if en_cell.get("cell_type") != "markdown":
            continue
        en_src = "".join(en_cell.get("source", []))
        ch_src = "".join(ch_cell.get("source", []))

        if should_skip(en_src):
            new_src = apply_headings(en_src)
        elif is_fully_translated(ch_src):
            new_src = apply_headings(ch_src)
        else:
            print(f"    Cell {idx}: translating...")
            new_src = translate_text(en_src, translator)

        if new_src != ch_src:
            ch_cell["source"] = source_to_list(new_src)
            changes += 1
    if changes:
        ch_path.write_text(json.dumps(ch_nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changes


def main():
    translator = None
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="en", target="zh-CN")
        print("Using Google Translate for incomplete cells")
    except ImportError:
        print("deep-translator not installed; using offline headings only")
        print("Install with: pip install deep-translator")

    total = 0
    for ipynb in sorted(CH07.rglob("*.ipynb")):
        if ipynb.name.endswith("_ch.ipynb"):
            continue
        ch_ipynb = ipynb.with_name(ipynb.stem + "_ch.ipynb")
        if not ch_ipynb.exists():
            continue
        print(f"  Notebook: {ipynb.relative_to(CH07)}")
        n = process_notebook(ipynb, ch_ipynb, translator)
        if n:
            print(f"  Updated {ch_ipynb.relative_to(CH07)}: {n} cells")
            total += n
        else:
            print(f"  No changes: {ch_ipynb.relative_to(CH07)}")
    print(f"\nDone: {total} cells updated")


if __name__ == "__main__":
    main()
