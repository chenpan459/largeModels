# 从零实现 Gemma 3 270M

本目录中的 [standalone-gemma3.ipynb](standalone-gemma3.ipynb) / [standalone-gemma3_ch.ipynb](standalone-gemma3_ch.ipynb) Jupyter notebook 包含 Gemma 3 270M 的从零实现，运行大约需要 2 GB 内存。

另一个 [standalone-gemma3-plus-kvcache.ipynb](standalone-gemma3-plus-kvcache.ipynb) / [standalone-gemma3-plus-kvcache_ch.ipynb](standalone-gemma3-plus-kvcache_ch.ipynb) notebook 增加了 KV cache 以提升运行时性能（但代码复杂度更高）。如需了解 KV cache，请参阅我的文章 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。

| 模型              | 模式              | 硬件            | Tokens/sec | GPU 显存 (VRAM) |
| ----------------- | ----------------- | --------------- | ---------- | --------------- |
| Gemma3Model 270M  | Regular           | Mac Mini M4 CPU | 8          | -               |
| Gemma3Model 270M  | Regular compiled  | Mac Mini M4 CPU | 9          | -               |
| Gemma3Model 270M  | KV cache          | Mac Mini M4 CPU | 130        | -               |
| Gemma3Model 270M  | KV cache compiled | Mac Mini M4 CPU | 224        | -               |
|                   |                   |                 |            |                 |
| Gemma3Model 270M  | Regular           | Mac Mini M4 GPU | 16         | -               |
| Gemma3Model 270M  | Regular compiled  | Mac Mini M4 GPU | Error      | -               |
| Gemma3Model 270M  | KV cache          | Mac Mini M4 GPU | 23         | -               |
| Gemma3Model 270M  | KV cache compiled | Mac Mini M4 GPU | Error      | -               |
|                   |                   |                 |            |                 |
| Gemma3Model 270M  | Regular           | Nvidia A100 GPU | 28         | 1.84 GB         |
| Gemma3Model 270M  | Regular compiled  | Nvidia A100 GPU | 128        | 2.12 GB         |
| Gemma3Model 270M  | KV cache          | Nvidia A100 GPU | 26         | 1.77 GB         |
| Gemma3Model 270M  | KV cache compiled | Nvidia A100 GPU | 99         | 2.12 GB         |


下方是与 Qwen3 0.6B 作为参考模型的并排对比；若对 Qwen3 0.6B 独立 notebook 感兴趣，可前往 [此处](../11_qwen3)。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gemma3/gemma3-vs-qwen3.webp">

<br>

如需了解架构差异并阅读与其他架构的对比，请参阅我的文章 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。


