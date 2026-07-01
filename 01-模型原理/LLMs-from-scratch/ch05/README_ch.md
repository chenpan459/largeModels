# 第 5 章：在无标注数据上预训练

&nbsp;
## 主章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主章节代码

&nbsp;
## 补充材料

- [02_alternative_weight_loading](02_alternative_weight_loading) 包含从 OpenAI 以外来源加载 GPT 模型权重的代码，以防 OpenAI 权重不可用
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) 包含在 Project Gutenberg 全部书籍语料上更长时间预训练 LLM 的代码
- [04_learning_rate_schedulers](04_learning_rate_schedulers) 包含更完善的训练函数实现，包括学习率调度器与梯度裁剪
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) 包含可选的超参数调优脚本
- [06_user_interface](06_user_interface) 实现与预训练 LLM 交互的界面
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) 包含 bonus notebook，展示如何更高效地通过 PyTorch 的 `load_state_dict` 加载模型权重
- [09_extending-tokenizers](09_extending-tokenizers) 包含 GPT-2 BPE 分词器的从零实现
- [10_llm-training-speed](10_llm-training-speed) 展示提升 LLM 训练速度的 PyTorch 性能技巧
- [18_muon](18_muon) 说明如何在 GPT 训练设置中使用 Muon 优化器

&nbsp;
## 从零实现 LLM 架构

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

&nbsp;


- [07_gpt_to_llama](07_gpt_to_llama) 包含将 GPT 架构实现逐步转换为 Llama 3.2 并加载 Meta AI 预训练权重的指南
- [11_qwen3](11_qwen3) 从零实现 Qwen3 0.6B 与 Qwen3 30B-A3B（混合专家），包括加载 base、reasoning 与 coding 等变体预训练权重的代码
- [12_gemma3](12_gemma3) 从零实现 Gemma 3 270M 及带 KV cache 的替代版本，包括加载预训练权重
- [13_olmo3](13_olmo3) 从零实现 Olmo 3 7B 与 32B（Base、Instruct、Think 变体）及带 KV cache 的替代版本，包括加载预训练权重
- [17_gemma4](17_gemma4) 从零实现 Gemma 4 的 E2B 与 E4B 稠密变体

&nbsp;
## 本章代码讲解视频

<br>
<br>

[![Link to the video](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)

## 中文文档

| 原文 | 中文版 |
|------|--------|
| [README.md](README.md) | [README_ch.md](README_ch.md) |
| 各子目录 `*.ipynb` | 对应 `*_ch.ipynb` |
