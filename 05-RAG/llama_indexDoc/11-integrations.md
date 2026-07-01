# 11 - Integrations（集成插件）

## 架构模式

LlamaIndex 采用 **「核心抽象 + 独立 PyPI 包」** 插件架构：

```
llama-index-core          # 定义 BaseLLM, BaseEmbedding, BasePydanticVectorStore…
llama-index-integrations/ # 每个集成一个子目录 → 独立包
```

每个集成包结构：

```
llama-index-llms-openai-like/
├── pyproject.toml
├── llama_index/llms/openai_like/
│   ├── __init__.py
│   └── base.py          # 继承 core Base 类
└── tests/
```

## 集成类别

`llama-index-integrations/` 顶层目录：

| 类别 | 示例包 | 继承基类 |
|------|--------|----------|
| `llms/` | openai, ollama, openai-like, llama-cpp, vllm | `LLM` |
| `embeddings/` | openai, openai-like, huggingface | `BaseEmbedding` |
| `vector_stores/` | qdrant, chroma, milvus, postgres | `BasePydanticVectorStore` |
| `readers/` | file, web, notion, database | `BaseReader` |
| `postprocessor/` | cohere-rerank, colbert-rerank | `BaseNodePostprocessor` |
| `tools/` | 各类外部 API | `AsyncBaseTool` |
| `graph_stores/` | neo4j, neptune | `GraphStore` |
| `storage/` | chat-store, kvstore | 各类 Store |
| `agent/` | anthropic, openai | Agent 扩展 |
| `voice_agents/` | openai, gemini-live | Voice |
| `retrievers/` | bm25, you | `BaseRetriever` |
| `programs/` | guidance, lmformatenforcer | 结构化输出 |
| `callbacks/` | langfuse, openinference | `BaseCallbackHandler` |

规模：**300+** 独立包（monorepo 内）。

## LLM 集成模式

### OpenAI 系

```python
from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-4o-mini")
```

### OpenAI 兼容（llama-server、vLLM、LocalAI）

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="my-model",
    api_base="http://127.0.0.1:8080/v1",
    api_key="fake",
    is_chat_model=True,
)
```

`OpenAILike` 继承 `OpenAI`，仅改 `api_base`，适配任意 OpenAI-compatible HTTP API。

### 本地推理

| 包 | 后端 |
|----|------|
| `llama-index-llms-llama-cpp` | llama.cpp Python binding |
| `llama-index-llms-ollama` | Ollama HTTP |
| `llama-index-llms-vllm` | vLLM |

### LlamaCPP

```python
from llama_index.llms.llama_cpp import LlamaCPP

llm = LlamaCPP(
    model_path="/path/to/model.gguf",
    context_window=4096,
)
```

直接加载 GGUF，**不经过** llama-server HTTP；与 kefu-kb 架构不同。

## Embedding 集成

```python
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

embed = OpenAILikeEmbedding(
    model_name="bge-m3",
    api_base="http://127.0.0.1:8081/v1",
    api_key="fake",
)
```

HuggingFace 本地：

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
```

## VectorStore 集成

```python
# pip install llama-index-vector-stores-qdrant
from llama_index.vector_stores.qdrant import QdrantVectorStore
```

常见：Qdrant、Chroma、Milvus、Weaviate、PGVector、Elasticsearch、Pinecone。

统一实现 `add` / `delete` / `query`，Index 层无感切换。

## Reader 集成

```python
from llama_index.core import SimpleDirectoryReader

# 扩展 reader 包
from llama_index.readers.file import PDFReader, MarkdownReader
```

`SimpleDirectoryReader` 在 core 中，自动按扩展名选 reader。

## Postprocessor / Rerank

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.sentence_transformers_rerank import SentenceTransformerRerank
```

## 安装约定

```bash
# 元包（含 OpenAI 默认）
pip install llama-index

# 按需安装
pip install llama-index-llms-openai-like
pip install llama-index-embeddings-openai-like
pip install llama-index-vector-stores-qdrant
```

monorepo 开发时用 `uv` / `pip install -e llama-index-integrations/llms/llama-index-llms-openai-like`。

## 编写自定义集成

1. 继承 core 对应 Base 类
2. 实现必要方法（如 `BaseEmbedding._get_text_embedding`）
3. 注册 `class_name()` 用于序列化
4. 可选：发布独立 PyPI 包

```python
from llama_index.core.base.embeddings.base import BaseEmbedding

class MyEmbedding(BaseEmbedding):
    @classmethod
    def class_name(cls) -> str:
        return "MyEmbedding"

    def _get_text_embedding(self, text: str) -> List[float]:
        ...
```

## 健康检查

monorepo 脚本 `scripts/integration_health_check.py` 用于 CI 检测各集成包导入与基本测试。

## 与本仓库推理栈的推荐组合

| 组件 | 推荐集成 | 对接 |
|------|----------|------|
| Chat | `OpenAILike` | llama-server :8080 |
| Embedding | `OpenAILikeEmbedding` | llama-server :8081 |
| Rerank | 自定义 Postprocessor 或 HTTP | llama-server rerank |
| VectorStore | `QdrantVectorStore` | kefu-kb docker compose |
| LLM 直连 GGUF | `LlamaCPP` | 03-推理部署/llama.cpp |

详见 [13-llama-cpp-integration.md](./13-llama-cpp-integration.md)。
