# 从零实现 Qwen3.5 0.8B

本文件夹包含 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 的从零实现风格代码。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/03.webp">

Qwen3.5 基于 Qwen3-Next 架构，我在文章 [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms) 的 [2.（线性）注意力混合架构](https://magazine.sebastianraschka.com/i/177848019/2-linear-attention-hybrids) 一节中有更详细的介绍。

<a href="https://magazine.sebastianraschka.com/p/beyond-standard-llms"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/02.webp" width="500px"></a>

请注意，Qwen3.5 在 `linear_attention` 与 `full_attention` 层之间交替。notebook 在保持完整模型流程可读的同时，复用了 [qwen3_5_transformers.py](qwen3_5_transformers.py) 中的线性注意力构建块；该文件包含 Hugging Face 的线性注意力代码，采用 Apache 2.0 开源许可。

&nbsp;
## 文件

- [qwen3.5.ipynb](qwen3.5.ipynb) / [qwen3.5_ch.ipynb](qwen3.5_ch.ipynb)：Qwen3.5 0.8B 主 notebook 实现。
- [qwen3.5-plus-kv-cache.ipynb](qwen3.5-plus-kv-cache.ipynb) / [qwen3.5-plus-kv-cache_ch.ipynb](qwen3.5-plus-kv-cache_ch.ipynb)：同一模型，含 KV cache 解码以提升效率。
- [qwen3_5_transformers.py](qwen3_5_transformers.py)：用于 Qwen3.5 线性注意力的 Hugging Face Transformers 辅助组件（保持英文，不翻译）。


