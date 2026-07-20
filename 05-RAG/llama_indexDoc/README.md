# LlamaIndex 0.14.23 中文源码导读

本目录是 `/home/cp/work2/largeModels/05-RAG/llama_index` 的配套文档中心，包含零基础教程和 01—19 源码分析。内容以本地源码快照为准，不以旧博客或其他版本的 API 反推。

## 版本与范围

| 项目 | 本文档范围 |
|---|---|
| `llama-index` umbrella 元包 | `0.14.23` |
| `llama-index-core` | `0.14.23`（根元包约束 `>=0.14.23,<0.15.0`） |
| `llama-index-instrumentation` | 本树 `0.5.0` |
| Python | 主包/core `>=3.10,<4.0` |
| 源码根目录 | `/home/cp/work2/largeModels/05-RAG/llama_index` |
| 文档根目录 | `/home/cp/work2/largeModels/05-RAG/llama_indexDoc` |

集成包有各自版本号，不能假定与 core 同步。本文分析的是 Python monorepo；TypeScript、LlamaCloud 服务端实现和外部数据库内部机制不在范围内。

## 按目标选择阅读路径

### 第一次接触 RAG

`小白 00 → 01 → 02 → 03 → 04 → 05 → 06`

先跑通最小 RAG，再阅读源码 01、02、03、06、07。

### 系统读懂一次 RAG 请求

`01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 12`

对应：项目布局 → 架构 → Node → Settings → ingestion → index → retrieval/query → synthesis → storage。

### 做生产 RAG

`05 → 07 → 08 → 12 → 15 → 16 → 19`

重点覆盖可重复摄取、查询调用链、持久化、离线评估、可观测与 CI。

### Agent / Workflow

`02 → 04 → 10 → 11 → 16`

先理解 core 与 Settings，再看 Agent/Workflow、集成边界和 instrumentation。

### 图 RAG

`03 → 05 → 12 → 17 → 15`

先理解 Node 与 StorageContext，再进入 PropertyGraphIndex，最后建立检索评估。

### 多模态与结构化提取

`03 → 06 → 07 → 18 → 15`

重点区分 ImageNode、ImageBlock、多模态检索和 Pydantic Program。

### 参与 monorepo 开发

`01 → 02 → 11 → 14 → 19`

了解包边界后，再使用 uv、llama-dev 和当前 CI/发布流程。

## 小白入门教程

| 文档 | 内容 |
|---|---|
| [00-什么是RAG.md](./00-小白入门/00-什么是RAG.md) | 用直观例子理解检索增强生成 |
| [01-安装与环境.md](./00-小白入门/01-安装与环境.md) | Python 环境、pip 包和测试数据 |
| [02-五分钟第一个RAG.md](./00-小白入门/02-五分钟第一个RAG.md) | 最小可运行 RAG |
| [03-核心概念对照表.md](./00-小白入门/03-核心概念对照表.md) | Document、Node、Index、Retriever |
| [04-Settings必知必会.md](./00-小白入门/04-Settings必知必会.md) | LLM、Embedding 与全局默认配置 |
| [05-读文档与切分.md](./00-小白入门/05-读文档与切分.md) | Reader、切分和中文参数建议 |
| [06-索引与查询.md](./00-小白入门/06-索引与查询.md) | 建索引、查询、持久化与异步 |
| [07-对接llama-server.md](./00-小白入门/07-对接llama-server.md) | 对接本地 llama.cpp 服务 |
| [08-常见报错与修复.md](./00-小白入门/08-常见报错与修复.md) | Key、连接、模型与依赖错误 |
| [09-进阶可选模块.md](./00-小白入门/09-进阶可选模块.md) | ChatEngine、Pipeline、Rerank 入门 |

## 源码分析 01—19

### 基础架构与数据链

| 编号 | 文档 | 核心问题 |
|---|---|---|
| 01 | [project-overview](./01-LlamaIndex项目总览.md) | Monorepo 有哪些包，core 与 integrations 如何分工 |
| 02 | [architecture](./02-整体架构与RAG数据流.md) | 分层架构和 RAG 主数据流 |
| 03 | [data-model](./03-数据模型与节点体系.md) | Document、BaseNode、TextNode、QueryBundle |
| 04 | [settings-config](./04-Settings配置机制.md) | Settings 的 lazy 默认与组件注入 |
| 05 | [ingestion-pipeline](./05-IngestionPipeline摄取流水线.md) | 文档转换、缓存、去重与写向量库 |
| 06 | [indices-vector-store](./06-索引与向量存储.md) | BaseIndex、VectorStoreIndex 与存储边界 |
| 07 | [retrieval-query-engine](./07-检索器与查询引擎.md) | Retriever → postprocessor → QueryEngine |
| 08 | [response-synthesizer](./08-响应合成器.md) | compact/refine/tree summarize 等合成策略 |

### 会话、Agent 与扩展

| 编号 | 文档 | 核心问题 |
|---|---|---|
| 09 | [chat-engine](./09-对话引擎与记忆.md) | 多轮上下文、condense 与 memory |
| 10 | [agent-workflow](./10-Agent与Workflow.md) | Agent、Tool、Workflow 事件模型 |
| 11 | [integrations](./11-集成插件体系.md) | 独立集成包的命名、依赖和加载边界 |
| 12 | [storage](./12-存储系统.md) | StorageContext、doc/index/vector/graph store |
| 13 | [llama-cpp-integration](./13-llama.cpp本地推理集成.md) | llama-server 与本地推理接入 |
| 14 | [api-reference](./14-API参考速查.md) | 常用导入、构建、查询与扩展基类速查 |

### 质量、观测与高级能力

| 编号 | 文档 | 核心问题 |
|---|---|---|
| 15 | [evaluation](./15-评估体系.md) | 检索 HitRate/MRR/NDCG 与生成质量评估 |
| 16 | [observability](./16-可观测性.md) | CallbackManager 与 instrumentation 双轨 |
| 17 | [property-graph](./17-PropertyGraph属性图.md) | 属性图构建、组合检索、存储与旧 KG 区别 |
| 18 | [multimodal-structured-output](./18-多模态与结构化输出.md) | ImageNode/Block、多模态查询和 Programs |
| 19 | [development-tooling](./19-Monorepo开发工具链.md) | uv、hatchling、llama-dev、CI 与发布 |

## 关键版本纠正

阅读旧教程时，优先检查以下差异：

1. **`ServiceContext` 不是当前主配置入口**：0.14.23 以 `Settings` 和显式组件参数为主。
2. **Agent 已迁移到 Workflow 风格**：不要照搬旧 `AgentRunner` 教程作为新项目架构。
3. **`KnowledgeGraphIndex` 已弃用**：自 0.10.53 起建议使用 `PropertyGraphIndex`；两者参数和存储模型不同。
4. **`SimpleMultiModalQueryEngine` 在 0.14.23 已弃用**：新代码使用标准 QueryEngine 的 `multimodal=True` 与 content-block prompt。
5. **可观测性是双轨**：`CallbackManager` 仍在工作，同时 core 通过独立 `llama-index-instrumentation` 发布强类型 Event/Span。
6. **core 下 instrumentation 是兼容重导出层的一部分**：真正 Dispatcher 实现在独立包 `llama-index-instrumentation/src/`。
7. **当前 CI 主测试不是只靠 Pants**：Makefile 保留 Pants target，但 PR Unit Test 使用 uv + `llama-dev test`。
8. **根 `llama-index` 是 umbrella 包**：大量 LLM、Embedding、VectorStore 是独立 PyPI 集成，不能只查 core 找实现。
9. **评估必须拆分检索与生成**：Retriever 指标按 Node ID，生成 evaluator 按 query/context/response，不能混成无解释总分。

## 源码地图

所有相对路径均相对于 `/home/cp/work2/largeModels/05-RAG/llama_index`：

```text
pyproject.toml
llama-index-core/
  pyproject.toml
  llama_index/core/
    schema.py                         # Document/Node/ImageNode/QueryBundle
    settings.py                       # Settings
    ingestion/pipeline.py             # IngestionPipeline
    indices/base.py                   # BaseIndex
    indices/vector_store/             # VectorStoreIndex/Retriever
    indices/property_graph/           # PropertyGraphIndex 与子 retriever
    indices/knowledge_graph/          # 已弃用 KnowledgeGraphIndex
    indices/multi_modal/              # 双向量库多模态索引
    base/base_retriever.py            # BaseRetriever
    query_engine/                     # QueryEngine
    response_synthesizers/            # 响应合成
    evaluation/                       # 检索/生成评估
    callbacks/                        # CallbackManager 轨道
    instrumentation/events/           # core 强类型业务事件
    program/                          # Pydantic Programs
    storage/storage_context.py        # 存储总装配
llama-index-instrumentation/
  src/llama_index_instrumentation/    # Dispatcher/Event/Span 真正实现
llama-index-integrations/             # LLM/Embedding/Store 等独立包
llama-index-core/pyproject.toml        # 依赖外部 llama-index-workflows 包
llama-dev/                            # monorepo CLI
.github/workflows/                    # 测试、lint、构建、发布
```

## 最小运行入口

```python
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine(similarity_top_k=3).query("问题")
print(response)
print(response.source_nodes)
```

在执行前显式配置 `Settings.llm` 和 `Settings.embed_model`；完整本地模型示例见小白教程 02、04、07 和源码文档 13。

## 上游

- 仓库：https://github.com/run-llama/llama_index
- 官方文档：https://docs.llamaindex.ai
- 许可证：MIT

当本文档与运行环境不一致时，按“已安装包版本 → 对应 tag 源码 → 集成包 pyproject”的顺序核对，不要直接套用 main 分支最新文档。
