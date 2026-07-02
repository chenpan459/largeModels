#!/usr/bin/env python3
"""Translate dpo-from-scratch.ipynb to Chinese dpo-from-scratch_ch.ipynb."""

import copy
import json
import re
import sys
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent
SRC = DIR / "dpo-from-scratch.ipynb"
TGT = DIR / "dpo-from-scratch_ch.ipynb"

HEADER_EN = (
    'Supplementary code for the <a href="http://mng.bz/orYv">Build a Large Language Model From Scratch</a> book by <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n'
    '<br>Code repository: <a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>'
)
HEADER_CH = (
    '<a href="http://mng.bz/orYv">《从零构建大语言模型》（Build a Large Language Model From Scratch）</a> 一书的配套代码，作者 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>\n'
    '<br>代码仓库：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>'
)

SECTION_HEADINGS = {
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

PRESERVE = [
    "GPT", "LLM", "LLMs", "PyTorch", "RLHF", "DPO", "JSON", "Python", "Jupyter",
    "Alpaca", "Ollama", "CUDA", "GPU", "CPU", "tiktoken", "PreferenceDataset",
    "InstructionDataset", "GPTModel", "DataLoader", "Google Colab", "Hugging Face",
    "Meta AI", "macOS", "Windows", "Linux", "REST", "API", "GitHub", "OpenAI",
    "Llama", "GPT-2", "GPT-4", "M3", "MacBook", "Safetensors", "Chainlit",
]

CODE_REPLACEMENTS = [
    ("# Tokenizer", "# 分词器"),
    ("# Deep learning library", "# 深度学习库"),
    ("# Pre-tokenize texts", "# 预先对文本分词"),
    ("# Initialize lists to hold batch data", "# 初始化用于存放 batch 数据的列表"),
    ("# Determine the longest sequence to set a common padding length", "# 确定最长序列以设置统一的 padding 长度"),
    ("# Process each item in the batch", "# 处理 batch 中的每个样本"),
    ("# Adjust padding according to the common maximum length", "# 按统一最大长度调整 padding"),
    ("# Set mask for all padding tokens to False", "# 将所有 padding token 的 mask 设为 False"),
    ("# Set mask for all input tokens to False", "# 将所有输入 token 的 mask 设为 False"),
    ('# +2 sets the 2 newline ("\\n") tokens before "### Response" to False', '# +2 将在 "### Response" 之前的 2 个换行符（"\\n"）token 设为 False'),
    ("# Final processing", "# 最终处理"),
    ("# Stack all sequences into a tensor for the given key", "# 将给定 key 的所有序列堆叠为张量"),
    ("# Optionally truncate to maximum sequence length", "# 可选：截断到最大序列长度"),
    ("# Move to the specified device", "# 移动到指定 device"),
    ("# Use PyTorch 2.9 or newer for stable mps results", "# 使用 PyTorch 2.9 或更高版本以获得稳定的 mps 结果"),
    ("device=device,            # Put the data directly on a GPU if available", "device=device,            # 若有 GPU 则直接将数据放到 GPU"),
    ("mask_prompt_tokens=True,  # This is optional", "mask_prompt_tokens=True,  # 可选"),
    ("allowed_max_length=1024   # The supported context length of the model", "allowed_max_length=1024   # 模型支持的上下文长度"),
    ("train_portion = int(len(data) * 0.85)  # 85% for training", "train_portion = int(len(data) * 0.85)  # 85% 用于训练"),
    ("test_portion = int(len(data) * 0.1)    # 10% for testing", "test_portion = int(len(data) * 0.1)    # 10% 用于测试"),
    ("val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation", "val_portion = len(data) - train_portion - test_portion  # 剩余 5% 用于验证"),
    ('print("Number of entries:", len(data))', 'print("条目数:", len(data))'),
    ('print("Training set length:", len(train_data))', 'print("训练集长度:", len(train_data))'),
    ('print("Validation set length:", len(val_data))', 'print("验证集长度:", len(val_data))'),
    ('print("Test set length:", len(test_data))', 'print("测试集长度:", len(test_data))'),
    ('print("chosen inputs:", batch["chosen"][0].shape)', 'print("chosen 输入:", batch["chosen"][0].shape)'),
    ('print("Train loader:")', 'print("训练 DataLoader:")'),
    ("# Try finding the model checkpoint locally:", "# 尝试在本地查找模型 checkpoint："),
    ("# If this notebook is run on Google Colab, get it from a Google Drive folder", "# 若在 Google Colab 中运行，从 Google Drive 文件夹获取"),
    ("google_drive_path = \"/content/drive/My Drive/Books/LLMs-From-Scratch/ch07/colab/gpt2-medium355M-sft.pth\"  # Readers need to adjust this path", "google_drive_path = \"/content/drive/My Drive/Books/LLMs-From-Scratch/ch07/colab/gpt2-medium355M-sft.pth\"  # 读者需调整此路径"),
    ('            f"Could not find \'{finetuned_model_path}\'.\\n"\n            "Run the `ch07.ipynb` notebook to finetune and save the finetuned model."', '            f"找不到 \'{finetuned_model_path}\'.\\n"\n            "请运行 `ch07_ch.ipynb` notebook 以微调并保存微调后的模型。"'),
    ("# If the `previous_chapters.py` file is not available locally,", "# 若本地没有 `previous_chapters.py` 文件，"),
    ("# you can import it from the `llms-from-scratch` PyPI package.", "# 可从 `llms-from-scratch` PyPI 包导入。"),
    ("# For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg", "# 详见：https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg"),
    ("# E.g.,", "# 例如："),
    ('    "vocab_size": 50257,     # Vocabulary size', '    "vocab_size": 50257,     # 词表大小'),
    ('    "context_length": 1024,  # Context length', '    "context_length": 1024,  # 上下文长度'),
    ('    "drop_rate": 0.0,        # Dropout rate', '    "drop_rate": 0.0,        # Dropout 比率'),
    ('    "qkv_bias": True         # Query-key-value bias', '    "qkv_bias": True         # QKV 偏置'),
    ("# Alternatively:", "# 或者："),
    ('    token_ids=batch["prompt"][0],  # [0] for the first entry in the batch', '    token_ids=batch["prompt"][0],  # [0] 表示 batch 中第一条'),
    ("    \"\"\"Compute the DPO loss for a batch of policy and reference model log probabilities.", "    \"\"\"计算一批 policy 与 reference 模型 log 概率的 DPO 损失。"),
    ("        model_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)", "        model_chosen_logprobs: policy 模型对 chosen 回复的 log 概率。形状：(batch_size,)"),
    ("        model_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)", "        model_rejected_logprobs: policy 模型对 rejected 回复的 log 概率。形状：(batch_size,)"),
    ("        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)", "        reference_chchosen_logprobs: reference 模型对 chosen 回复的 log 概率。形状：(batch_size,)"),
]

def protect_terms(text):
    mapping = {}
    result = text
    for i, term in enumerate(PRESERVE):
        ph = f"__KEEP_{i}__"
        if term in result:
            mapping[ph] = term
            result = result.replace(term, ph)
    return result, mapping

def restore_terms(text, mapping):
    for ph, term in mapping.items():
        text = text.replace(ph, term)
    return text

def apply_post(text):
    text = text.replace(HEADER_EN, HEADER_CH)
    for en, ch in SECTION_HEADINGS.items():
        text = text.replace(en, ch)
    text = text.replace("ch07.ipynb", "ch07_ch.ipynb")
    text = text.replace("create-preference-data-ollama.ipynb", "create-preference-data-ollama_ch.ipynb")
    return text

def should_skip_md(text):
    s = text.strip()
    return (not s) or s == "&nbsp;" or s.startswith("<table")

def translate_md(text, translator):
    protected, mapping = protect_terms(text)
    chunks, cur = [], ""
    for line in protected.split("\n"):
        if len(cur) + len(line) + 1 > 4500:
            if cur:
                chunks.append(cur)
            cur = line
        else:
            cur = cur + "\n" + line if cur else line
    if cur:
        chunks.append(cur)
    parts = []
    for ch in chunks:
        parts.append(ch if not ch.strip() else translator.translate(ch))
        time.sleep(0.05)
    return restore_terms("\n".join(parts), mapping)

def to_source_list(text):
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]

def translate_code(src):
    out = src
    for a, b in CODE_REPLACEMENTS:
        out = out.replace(a, b)
    return out

def clear_cell(cell):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

def main():
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("Install: pip install deep-translator", file=sys.stderr)
        sys.exit(1)
    translator = GoogleTranslator(source="en", target="zh-CN")
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    out = copy.deepcopy(nb)
    for idx, cell in enumerate(out["cells"]):
        if cell["cell_type"] == "markdown":
            en = "".join(cell.get("source", []))
            if should_skip_md(en):
                zh = en.replace(HEADER_EN, HEADER_CH) if en.strip().startswith("<table") else en
            else:
                print(f"Translating markdown cell {idx}...")
                zh = translate_md(en, translator)
            cell["source"] = to_source_list(apply_post(zh))
        elif cell["cell_type"] == "code":
            cell["source"] = to_source_list(translate_code("".join(cell.get("source", []))))
        clear_cell(cell)
    TGT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cn_md = sum(1 for c in out["cells"] if c["cell_type"] == "markdown" and re.search(r"[\u4e00-\u9fff]", "".join(c["source"])))
    print(f"Wrote {TGT.name}: {len(out['cells'])} cells, {cn_md} Chinese markdown cells")

if __name__ == "__main__":
    main()
