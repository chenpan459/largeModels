# LlamaIndex 项目文档

本目录包含对 `/home/cp/work2/largeModels/05-RAG/llama_index` 项目的结构化源码分析文档。

## 文档索引

### 概览与架构

| 文档 | 说明 |
|------|------|
| [01-project-overview.md](./01-project-overview.md) | Monorepo 总览、版本与目录结构 |
| [02-architecture.md](./02-architecture.md) | 分层架构与 RAG 数据流 |

### 核心模块

| 文档 | 说明 |
|------|------|
| [03-data-model.md](./03-data-model.md) | Document / Node / QueryBundle 数据模型 |
| [04-settings-config.md](./04-settings-config.md) | Settings 全局配置 |
| [05-ingestion-pipeline.md](./05-ingestion-pipeline.md) | 文档摄取、切分与变换 |
| [06-indices-vector-store.md](./06-indices-vector-store.md) | Index 与 VectorStore |
| [07-retrieval-query-engine.md](./07-retrieval-query-engine.md) | Retriever 与 QueryEngine |
| [08-response-synthesizer.md](./08-response-synthesizer.md) | 响应合成策略 |
| [09-chat-engine.md](./09-chat-engine.md) | 多轮对话 ChatEngine |
| [10-agent-workflow.md](./10-agent-workflow.md) | Agent 与 Workflow |
| [11-integrations.md](./11-integrations.md) | 300+ 集成插件模式 |
| [12-storage.md](./12-storage.md) | StorageContext 与持久化 |

### 实践与参考

| 文档 | 说明 |
|------|------|
| [13-llama-cpp-integration.md](./13-llama-cpp-integration.md) | 对接 llama-server / kefu-kb |
| [14-api-reference.md](./14-api-reference.md) | 常用 API 速查 |

## 项目路径

```
/home/cp/work2/largeModels/05-RAG/llama_index/
├── llama-index-core/          # 核心框架 (~475 个 core 模块文件)
├── llama-index-integrations/    # LLM / Embedding / VectorStore 等插件
├── llama-index-utils/           # 辅助工具包
└── docs/                        # 官方文档源
```

## 推荐阅读顺序

1. **入门 RAG**：01 → 02 → 03 → 06 → 07
2. **生产摄取**：04 → 05 → 12
3. **对话与 Agent**：08 → 09 → 10
4. **本地部署**：11 → 13（配合 `03-推理部署/llama.cpp`）
5. **业务落地**：13 → `07-业务应用/kefu-kb/`

## 快速参考

```python
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core import SimpleDirectoryReader

# 读取 → 建索引 → 查询
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("你的问题")
print(response)
```

```bash
# 安装（元包，含 OpenAI 默认集成）
pip install llama-index

# 仅核心
pip install llama-index-core
```

## 版本信息

| 包 | 版本 |
|----|------|
| llama-index (元包) | 0.14.23 |
| llama-index-core | ≥0.14.23, <0.15.0 |

## 上游项目

- 仓库: https://github.com/run-llama/llama_index
- 许可证: MIT
- 官方文档: https://docs.llamaindex.ai
