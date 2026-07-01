#!/usr/bin/env python3
"""Offline markdown translation for ch05 *_ch.ipynb files (no network required)."""

import json
import re
import sys
from pathlib import Path

CH05 = Path(__file__).resolve().parent

HEADER_EN = (
    'Supplementary code for the <a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> book by <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n'
    '<br>Code repository: <a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>'
)
HEADER_CH = (
    '<a href="http://mng.bz/orYv">《从零构建大语言模型》（Build a Large Language Model From Scratch）</a> 一书的配套代码，作者 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n'
    '<br>代码仓库：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>'
)

# Longest-first phrase replacements for markdown prose
PHRASES = [
    ("Supplementary code for the", "《从零构建大语言模型》配套代码："),
    ("Code repository:", "代码仓库："),
    ("Bonus Code for Chapter 5", "第 5 章补充代码"),
    ("Bonus code for chapter 5", "第 5 章补充代码"),
    ("Alternative Weight Loading from PyTorch state dicts", "从 PyTorch state dict 加载权重的替代方案"),
    ("Alternative weight loading from Hugging Face Transformers", "从 Hugging Face Transformers 加载权重的替代方案"),
    ("Alternative weight loading from Hugging Face safetensors", "从 Hugging Face safetensors 加载权重的替代方案"),
    ("In the main chapter, we loaded the GPT model weights directly from OpenAI",
     "在主章节中，我们直接从 OpenAI 加载 GPT 模型权重"),
    ("This notebook provides alternative weight loading code to load the model weights from PyTorch state dict files that I created from the original TensorFlow files and uploaded to the",
     "本 notebook 提供替代权重加载代码，从我从原始 TensorFlow 文件创建并上传到"),
    ("This is conceptually the same as loading weights of a PyTorch model from via the state-dict method described in chapter 5:",
     "这在概念上与第 5 章描述的通过 state-dict 方法加载 PyTorch 模型权重相同："),
    ("Choose model", "选择模型"),
    ("Download file", "下载文件"),
    ("Load weights", "加载权重"),
    ("Generate text", "生成文本"),
    ("Converting a From-Scratch GPT Architecture to Llama 2",
     "将从零实现的 GPT 架构转换为 Llama 2"),
    ("Converting Llama 2 to Llama 3, Llama 3.1, and Llama 3.2",
     "将 Llama 2 转换为 Llama 3、Llama 3.1 和 Llama 3.2"),
    ("Standalone Llama 3.2", "独立 Llama 3.2 实现"),
    ("In this notebook, we convert", "在本 notebook 中，我们将"),
    ("step by step", "逐步"),
    ("from scratch", "从零实现"),
    ("From Scratch", "从零实现"),
    ("from-scratch", "从零实现"),
    ("Note that", "注意，"),
    ("Note:", "注意："),
    ("Tip:", "提示："),
    ("Warning:", "警告："),
    ("However,", "不过，"),
    ("Furthermore,", "此外，"),
    ("Additionally,", "另外，"),
    ("Specifically,", "具体而言，"),
    ("Alternatively,", "或者，"),
    ("Lastly,", "最后，"),
    ("First,", "首先，"),
    ("Next,", "接下来，"),
    ("Then,", "然后，"),
    ("Finally,", "最后，"),
    ("For example,", "例如，"),
    ("For instance,", "例如，"),
    ("In practice,", "实践中，"),
    ("In this section,", "在本节中，"),
    ("In the following", "在以下"),
    ("The following", "以下"),
    ("We can", "我们可以"),
    ("We will", "我们将"),
    ("We use", "我们使用"),
    ("We need", "我们需要"),
    ("Let's", "让我们"),
    ("This means", "这意味着"),
    ("This allows", "这允许"),
    ("This ensures", "这确保"),
    ("This section", "本节"),
    ("This notebook", "本 notebook"),
    ("This folder", "本文件夹"),
    ("This code", "此代码"),
    ("This is", "这是"),
    ("This folder contains", "本文件夹包含"),
    ("The main chapter", "主章节"),
    ("pretrained weights", "预训练权重"),
    ("Pretrained weights", "预训练权重"),
    ("weight loading", "权重加载"),
    ("Weight loading", "权重加载"),
    ("training loop", "训练循环"),
    ("Training loop", "训练循环"),
    ("text generation", "文本生成"),
    ("Text generation", "文本生成"),
    ("tokenizer", "分词器"),
    ("Tokenizer", "分词器"),
    ("attention mechanism", "注意力机制"),
    ("Attention mechanism", "注意力机制"),
    ("multi-head attention", "多头注意力"),
    ("Multi-head attention", "多头注意力"),
    ("feed forward", "前馈"),
    ("Feed forward", "前馈"),
    ("layer normalization", "层归一化"),
    ("Layer normalization", "层归一化"),
    ("embedding layer", "嵌入层"),
    ("Embedding layer", "嵌入层"),
    ("output layer", "输出层"),
    ("Output layer", "输出层"),
    ("context length", "上下文长度"),
    ("Context length", "上下文长度"),
    ("vocabulary size", "词汇表大小"),
    ("Vocabulary size", "词汇表大小"),
    ("dropout rate", "Dropout 比率"),
    ("Dropout rate", "Dropout 比率"),
    ("learning rate", "学习率"),
    ("Learning rate", "学习率"),
    ("gradient clipping", "梯度裁剪"),
    ("Gradient clipping", "梯度裁剪"),
    ("loss function", "损失函数"),
    ("Loss function", "损失函数"),
    ("validation loss", "验证损失"),
    ("Validation loss", "验证损失"),
    ("training loss", "训练损失"),
    ("Training loss", "训练损失"),
    ("inference", "推理"),
    ("Inference", "推理"),
    ("pretraining", "预训练"),
    ("Pretraining", "预训练"),
    ("finetuning", "微调"),
    ("Finetuning", "微调"),
    ("checkpoint", "检查点"),
    ("Checkpoint", "检查点"),
    ("state dict", "state dict"),
    ("State dict", "State dict"),
    ("memory efficient", "内存高效"),
    ("Memory efficient", "内存高效"),
    ("KV cache", "KV 缓存"),
    ("Mixture-of-Experts", "混合专家（MoE）"),
    ("mixture of experts", "混合专家"),
    ("Mixture of experts", "混合专家"),
    ("Grouped Query Attention", "分组查询注意力（GQA）"),
    ("grouped query attention", "分组查询注意力"),
    ("Rotary Position Embedding", "旋转位置嵌入（RoPE）"),
    ("rotary position embedding", "旋转位置嵌入"),
    ("sliding window attention", "滑动窗口注意力"),
    ("Sliding window attention", "滑动窗口 attention"),
    ("special tokens", "特殊 token"),
    ("Special tokens", "特殊 token"),
    ("new tokens", "新 token"),
    ("New tokens", "新 token"),
    ("Exercise", "练习"),
    ("Solution", "解答"),
    ("Optional", "可选"),
    ("Recommended", "推荐"),
    ("Implementation", "实现"),
    ("implementation", "实现"),
    ("Overview", "概述"),
    ("Summary", "小结"),
    ("Introduction", "简介"),
    ("Conclusion", "结论"),
    ("Comparison", "对比"),
    ("Performance", "性能"),
    ("Benchmark", "基准测试"),
    ("Experiment", "实验"),
    ("Results", "结果"),
    ("Setup", "设置"),
    ("Installation", "安装"),
    ("Requirements", "依赖"),
    ("Usage", "用法"),
    ("Example", "示例"),
    ("Examples", "示例"),
]

SECTION_HEADINGS = {
    "# Chapter 5: Pretraining on Unlabeled Data": "# 第 5 章：在无标注数据上预训练",
    "## 5.1 Evaluating generative text models": "## 5.1 评估生成式文本模型",
    "## 5.2 Training an LLM": "## 5.2 训练 LLM",
    "## 5.3 Decoding strategies to control randomness": "## 5.3 控制随机性的解码策略",
    "## 5.4 Loading and saving weights in PyTorch": "## 5.4 在 PyTorch 中加载和保存权重",
    "## 5.5 Loading pretrained weights from OpenAI": "## 5.5 从 OpenAI 加载预训练权重",
    "### Choose model": "### 选择模型",
    "### Download file": "### 下载文件",
    "### Load weights": "### 加载权重",
    "### Generate text": "### 生成文本",
    "### Exercise solutions": "### 练习解答",
}


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
    s = text.strip()
    if not s:
        return True
    if s.startswith("```") or s.startswith("<img") or s.startswith("!["):
        return True
    if s.startswith("http") or s.startswith("Ep ") or s.startswith("Time:"):
        return True
    if "Every effort moves" in s:
        return True
    return False


def translate_markdown(text: str, en_text: str | None = None) -> str:
    if should_skip(text):
        return text
    if chinese_ratio(text) >= 0.55:
        return text

    result = text
    if HEADER_EN in result:
        result = result.replace(HEADER_EN, HEADER_CH)

    for en, ch in SECTION_HEADINGS.items():
        result = result.replace(en, ch)

    # Apply phrase replacements (longest first already sorted)
    for en, ch in PHRASES:
        result = result.replace(en, ch)

    # If still mostly English and we have English source, use source with same replacements
    if chinese_ratio(result) < 0.35 and en_text and en_text != text:
        result = en_text
        if HEADER_EN in result:
            result = result.replace(HEADER_EN, HEADER_CH)
        for en, ch in SECTION_HEADINGS.items():
            result = result.replace(en, ch)
        for en, ch in PHRASES:
            result = result.replace(en, ch)

    return result


def source_to_lines(text: str, template) -> list:
    if isinstance(template, list):
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n") and template and template[-1].endswith("\n"):
            lines[-1] += "\n"
        return lines if lines else [text]
    return text


def process_notebook(en_path: Path, ch_path: Path) -> int:
    with open(en_path, encoding="utf-8") as f:
        en_nb = json.load(f)
    if ch_path.exists():
        with open(ch_path, encoding="utf-8") as f:
            ch_nb = json.load(f)
    else:
        ch_nb = json.loads(json.dumps(en_nb))

    en_cells = en_nb.get("cells", [])
    ch_cells = ch_nb.get("cells", [])
    if len(en_cells) != len(ch_cells):
        ch_cells = json.loads(json.dumps(en_cells))
        ch_nb["cells"] = ch_cells

    changes = 0
    for i, (en_cell, ch_cell) in enumerate(zip(en_cells, ch_cells)):
        if en_cell.get("cell_type") != "markdown":
            continue
        en_src = "".join(en_cell.get("source", []))
        ch_src = "".join(ch_cell.get("source", []))
        new_src = translate_markdown(ch_src, en_src)
        if new_src != ch_src:
            ch_cell["source"] = source_to_lines(new_src, en_cell.get("source", []))
            changes += 1

    if changes:
        with open(ch_path, "w", encoding="utf-8") as f:
            json.dump(ch_nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
    return changes


def main():
    total_files = 0
    total_cells = 0
    for ipynb in sorted(CH05.rglob("*.ipynb")):
        if ipynb.name.endswith("_ch.ipynb"):
            continue
        ch_ipynb = ipynb.with_name(ipynb.stem + "_ch.ipynb")
        if not ch_ipynb.exists():
            continue
        n = process_notebook(ipynb, ch_ipynb)
        if n:
            print(f"Updated {ch_ipynb.relative_to(CH05)}: {n} cells")
            total_files += 1
            total_cells += n
    print(f"\nDone: {total_files} notebooks, {total_cells} cells updated")


if __name__ == "__main__":
    main()
