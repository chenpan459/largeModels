# 03 - 推理部署

本地推理、GGUF 格式、HTTP 服务、高吞吐 serving。

| 项目 | 说明 | 入口 |
|------|------|------|
| [llama.cpp](llama.cpp/) | C/C++ 推理框架，量化、多后端 | `include/llama.h`, `tools/server/` |
| [llama.cppDoc](llama.cppDoc/) | llama.cpp 中文源码分析 | `README.md` 文档索引 |
| [vllm](vllm/) | PagedAttention，生产级 serving | `vllm/entrypoints/` |
| [vllmDoc](vllmDoc/) | vLLM 中文源码分析（12 篇） | [README](vllmDoc/README.md) |

**llama-server** 位于 `llama.cpp/tools/server/`，提供 OpenAI 兼容 API。

**学习顺序**: llama.cppDoc -> llama.cpp 源码 -> [vllmDoc](vllmDoc/README.md)
