# 从零实现 Olmo 3 7B 与 32B

本目录中的 [standalone-olmo3.ipynb](standalone-olmo3.ipynb) / [standalone-olmo3_ch.ipynb](standalone-olmo3_ch.ipynb) Jupyter notebook 包含 Olmo 3 7B 与 32B 的从零实现，运行大约需要 13 GB 内存。

另一个 [standalone-olmo3-plus-kv-cache.ipynb](standalone-olmo3-plus-kv-cache.ipynb) / [standalone-olmo3-plus-kv-cache_ch.ipynb](standalone-olmo3-plus-kv-cache_ch.ipynb) notebook 增加了 KV cache 以提升运行时性能（但代码复杂度更高）。如需了解 KV cache，请参阅我的文章 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。

下方是与 Qwen3 作为参考模型的并排对比；若对 Qwen3 0.6B 独立 notebook 感兴趣，可前往 [此处](../11_qwen3)。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-7B.webp?1">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-32B.webp?1">

Olmo 3 还有多种变体（如下所示；架构相同，仅训练流程不同）：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-pipeline.webp?1">


&nbsp;
## Olmo 3 与 Qwen3 的对比

本节聚焦架构（而非训练细节），简要对比 Olmo 3 与 Qwen3。


**7B 模型：**

1. 如上所示，Olmo 3 架构与 Qwen3 较为相似。不过值得注意的是，这很可能主要受 Olmo 2 前代启发，而非 Qwen3。

2. 与 Olmo 2 类似，Olmo 3 仍采用 post-norm 风格而非 pre-norm，因为 Olmo 2 论文发现这有助于稳定训练。

3. 有趣的是，7B 模型仍使用与 Olmo 2 类似的多头注意力（MHA）。不过为提升效率并减小 KV cache 体积，现已采用滑动窗口注意力（例如与 Gemma 3 类似）。

**32B 模型：**

4. 整体架构相同，只是规模更大。此外，各层比例（例如前馈层从输入到中间维度的扩展等）与 Qwen3 大致相当。

5. 我猜测，由于词表较小，架构最初略小于 Qwen3；随后他们将 Qwen3 中 5× 的中间维度扩展比例提高到 Olmo 3 的 5.4×，从而得到 32B 模型以便直接对比。

6. 另外请注意，32B 模型（终于！）使用了分组查询注意力（GQA）。





<br>

如需了解架构差异并阅读与其他架构的对比，请参阅我的文章 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) 文章。



