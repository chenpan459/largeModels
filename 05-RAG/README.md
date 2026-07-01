# 05 - RAG

检索增强生成、知识库、向量检索编排。

| 项目 | 说明 | 入口 |
|------|------|------|
| [llama_index](llama_index/) | 数据索引、检索、Agent 框架 | `docs/`, `llama-index-core/` |
| [llama_indexDoc](llama_indexDoc/) | LlamaIndex 源码分析文档（14 篇） | [README](llama_indexDoc/README.md) |

**配合使用**: `03-推理部署/llama.cpp` 的 llama-server 提供 embedding / rerank / chat API。

**学习顺序**: [llama_indexDoc](llama_indexDoc/README.md) → 对接 llama-server → [kefu-kb](../07-业务应用/kefu-kb/) 业务实践
