# 第 4 章：从零实现 GPT 模型以生成文本

&nbsp;
## 主章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主章节代码

&nbsp;
## 补充材料

- [02_performance-analysis](02_performance-analysis) 包含可选代码，分析主章节实现的 GPT 模型的性能
- [03_kv-cache](03_kv-cache) 实现 KV 缓存以加速推理时的文本生成
- [07_moe](07_moe) 混合专家（MoE）的说明与实现
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama) 包含将 GPT 架构实现转换为 Llama 3.2 并逐步加载 Meta AI 预训练权重的指南（完成第 4 章后了解替代架构可能很有意思，也可留到第 5 章后再看）


&nbsp;
## 注意力机制替代方案

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

&nbsp;

- [04_gqa](04_gqa) 介绍分组查询注意力（GQA），大多数现代 LLM（Llama 4、gpt-oss、Qwen3、Gemma 3 等）用作常规多头注意力（MHA）的替代
- [05_mla](05_mla) 介绍多头潜在注意力（MLA），DeepSeek V3 用作常规 MHA 的替代
- [06_swa](06_swa) 介绍滑动窗口注意力（SWA），Gemma 3 等模型使用
- [08_deltanet](08_deltanet) 介绍 Gated DeltaNet，一种流行的线性注意力变体（Qwen3-Next 和 Kimi Linear 使用）
- [10_kv-sharing](10_kv-sharing) 介绍跨层 KV 共享，Gemma 4 E2B 和 E4B 用于减少 KV 缓存内存


&nbsp;
## 更多

下方视频为章节内容的代码讲解补充。

<br>
<br>

[![Link to the video](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)

## 中文文档

| 原文 | 中文版 |
|------|--------|
| [README.md](README.md) | [README_ch.md](README_ch.md) |
| 各子目录 `*.ipynb` | 对应 `*_ch.ipynb` |
