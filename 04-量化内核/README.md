# 04 - 量化 / 内核

张量计算引擎、量化格式、CPU/GPU kernel 优化。

| 项目 | 说明 | 入口 |
|------|------|------|
| [ggml](ggml/) | llama.cpp 底层计算库（独立仓库） | `src/ggml-quants.c`, `src/ggml-cuda/` |

**关联**: llama.cpp 内嵌 GGUF 子模块在 `03-推理部署/llama.cpp/ggml/`，可与本目录对照阅读。

**学习顺序**: ggml 量化类型 -> CUDA/CPU kernel -> llama.cpp 中的调用
