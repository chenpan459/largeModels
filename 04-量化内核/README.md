# 04 - 量化 / 内核

张量计算引擎、量化格式、CPU/GPU kernel 优化。

| 项目 | 说明 | 入口 |
|------|------|------|
| [ggml](ggml/) | llama.cpp 底层计算库（GGML 0.15.3） | `src/ggml.c`, `src/ggml-quants.c` |
| [ggmlDoc](ggmlDoc/) | **GGML 中文源码分析文档** | [README.md](ggmlDoc/README.md) |

## 文档索引（ggmlDoc）

| 文档 | 说明 |
|------|------|
| [01-project-overview](ggmlDoc/01-project-overview.md) | 项目总览 |
| [02-architecture](ggmlDoc/02-architecture.md) | 分层架构 |
| [03-tensor-graph](ggmlDoc/03-tensor-graph.md) | 张量与计算图 |
| [04-backend-scheduler](ggmlDoc/04-backend-scheduler.md) | Backend 调度器 |
| [05-memory-alloc](ggmlDoc/05-memory-alloc.md) | 内存分配器 |
| [06-quantization](ggmlDoc/06-quantization.md) | 量化系统 |
| [07-gguf-format](ggmlDoc/07-gguf-format.md) | GGUF 格式 |
| [08-backend-cpu](ggmlDoc/08-backend-cpu.md) | CPU Backend |
| [09-backend-gpu](ggmlDoc/09-backend-gpu.md) | CUDA/Metal/Vulkan |
| [13-llama-cpp-integration](ggmlDoc/13-llama-cpp-integration.md) | 与 llama.cpp 集成 |

**关联**: llama.cpp 内嵌副本在 `03-推理部署/llama.cpp/ggml/`，中文分析见 `03-推理部署/llama.cppDoc/`。

**学习顺序**: ggmlDoc 01-03 -> 06-07 -> 04-05 -> 08-09 -> 13
