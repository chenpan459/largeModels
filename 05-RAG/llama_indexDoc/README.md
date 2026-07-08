# LlamaIndex 项目文档

本目录包含对 `/home/cp/work2/largeModels/05-RAG/llama_index` 的源码分析与 **小白入门教程**。

## 你是谁？选哪条路

| 你是… | 从这里开始 |
|--------|-----------|
| **第一次接触 RAG** | 👉 [00-小白入门/](./00-小白入门/) 系列（推荐） |
| **已跑通 RAG，想读源码** | 01 → 02 → 03 → 06 → 07 |
| **要做生产部署** | 05 → 12 → 13 + kefu-kb |
| **查 API** | [14-api-reference.md](./14-api-reference.md) |

---

## 小白入门（从零学 RAG + LlamaIndex）

| 文档 | 说明 |
|------|------|
| [00-什么是RAG.md](./00-小白入门/00-什么是RAG.md) | 5 分钟弄懂检索增强生成 |
| [01-安装与环境.md](./00-小白入门/01-安装与环境.md) | pip 包、虚拟环境、测试数据 |
| [02-五分钟第一个RAG.md](./00-小白入门/02-五分钟第一个RAG.md) | **完整可运行脚本** |
| [03-核心概念对照表.md](./00-小白入门/03-核心概念对照表.md) | Document、Node、Index 通俗解释 |
| [04-Settings必知必会.md](./00-小白入门/04-Settings必知必会.md) | 为什么必须先配 Settings |
| [05-读文档与切分.md](./00-小白入门/05-读文档与切分.md) | Reader、chunk_size 中文建议 |
| [06-索引与查询.md](./00-小白入门/06-索引与查询.md) | persist、Qdrant、async |
| [07-对接llama-server.md](./00-小白入门/07-对接llama-server.md) | 本地 llama.cpp 双端口 |
| [08-常见报错与修复.md](./00-小白入门/08-常见报错与修复.md) | OPENAI_API_KEY、Connection refused 等 |
| [09-进阶可选模块.md](./00-小白入门/09-进阶可选模块.md) | ChatEngine、Pipeline、Rerank |

**建议 3 天路径**：Day1 `00→02` | Day2 `04→07` + 实操 | Day3 `08→09` + 进阶文档

---

## 源码分析文档（进阶）

### 概览与架构

| 文档 | 说明 |
|------|------|
| [01-project-overview.md](./01-project-overview.md) | Monorepo、模块一览 |
| [02-architecture.md](./02-architecture.md) | 分层架构与 RAG 数据流 |

### 核心模块

| 文档 | 说明 |
|------|------|
| [03-data-model.md](./03-data-model.md) | Document / Node / QueryBundle |
| [04-settings-config.md](./04-settings-config.md) | Settings 全局配置 |
| [05-ingestion-pipeline.md](./05-ingestion-pipeline.md) | 摄取流水线 |
| [06-indices-vector-store.md](./06-indices-vector-store.md) | Index 与 VectorStore |
| [07-retrieval-query-engine.md](./07-retrieval-query-engine.md) | Retriever 与 QueryEngine |
| [08-response-synthesizer.md](./08-response-synthesizer.md) | 响应合成策略 |
| [09-chat-engine.md](./09-chat-engine.md) | 多轮 ChatEngine |
| [10-agent-workflow.md](./10-agent-workflow.md) | Agent 与 Workflow |
| [11-integrations.md](./11-integrations.md) | 300+ 集成插件 |
| [12-storage.md](./12-storage.md) | StorageContext 持久化 |

### 实践与参考

| 文档 | 说明 |
|------|------|
| [13-llama-cpp-integration.md](./13-llama-cpp-integration.md) | llama-server / kefu-kb |
| [14-api-reference.md](./14-api-reference.md) | API 速查 |

---

## 项目路径

```
/home/cp/work2/largeModels/05-RAG/llama_index/
├── llama-index-core/           # 核心框架
├── llama-index-integrations/   # LLM / Embedding / VectorStore 插件
├── llama-index-utils/
└── docs/

/home/cp/work2/largeModels/05-RAG/llama_indexDoc/   # 本文档
```

---

## 最快跑通（复制即用）

```python
# 见 00-小白入门/02-五分钟第一个RAG.md 完整版
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
# ... 先 Settings，再 from_documents，再 query
```

```bash
pip install llama-index-core llama-index-llms-openai-like \
  llama-index-embeddings-openai-like
```

---

## 与本仓库其他模块

| 模块 | 关系 |
|------|------|
| `07-业务应用/kefu-kb` | 自研 RAG，可迁移为 LlamaIndex |
| `03-推理部署/llama.cpp` | llama-server 作 LLM/Embedding 后端 |
| `03-推理部署/vllm` | 亦可用 OpenAILike 对接 |

---

## 版本与上游

| 包 | 版本 |
|----|------|
| llama-index（元包） | 0.14.23 |
| llama-index-core | ≥0.14.23, <0.15.0 |

- 仓库：https://github.com/run-llama/llama_index
- 官方文档：https://docs.llamaindex.ai
- 许可证：MIT
