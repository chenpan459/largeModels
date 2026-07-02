# Gemma 4

本目录包含基于 Gemma 3 参考 notebook 构建、并适配稠密 `google/gemma-4-E2B` 与 `google/gemma-4-E4B` 检查点的独立纯文本 Gemma 4 notebook。

- [standalone-gemma4.ipynb](./standalone-gemma4.ipynb) / [standalone-gemma4_ch.ipynb](./standalone-gemma4_ch.ipynb)：在纯 PyTorch 中实现共享的 Gemma 4 稠密架构，并通过 `CHOOSE_MODEL` 在 E2B 与 E4B 参考配置之间切换。
- [standalone-gemma4-plus-kvcache.ipynb](./standalone-gemma4-plus-kvcache.ipynb) / [standalone-gemma4-plus-kvcache_ch.ipynb](./standalone-gemma4-plus-kvcache_ch.ipynb)：同一模型，含 KV cache 解码以提升效率。


