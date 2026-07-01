# 14 - API 速查

## 顶层导入（llama_index.core）

```python
from llama_index.core import (
    # 数据
    Document,
    Settings,
    StorageContext,
    load_index_from_storage,
    # 索引
    VectorStoreIndex,
    SummaryIndex,
    # 读取
    SimpleDirectoryReader,
    # 响应
    PromptTemplate,
)
```

## Settings

```python
Settings.llm = ...
Settings.embed_model = ...
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
Settings.callback_manager = CallbackManager([handler])
Settings.transformations = [Settings.node_parser]
```

## 索引 CRUD

```python
# 创建
index = VectorStoreIndex.from_documents(docs)
index = VectorStoreIndex.from_vector_store(vector_store=vs)
index = VectorStoreIndex(nodes)

# 追加
index.insert(document)
index.insert_nodes(nodes)

# 删除
index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)

# 持久化
index.storage_context.persist("./storage")
index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))
```

## 检索与查询

```python
# Retriever
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("query")

# QueryEngine
engine = index.as_query_engine(similarity_top_k=5, response_mode="compact")
response = engine.query("query")
await engine.aquery("query")

# 流式
engine = index.as_query_engine(streaming=True)
for token in engine.query("q").response_gen:
    ...
```

## ChatEngine

```python
chat = index.as_chat_engine(chat_mode="condense_plus_context")
response = chat.chat("message")
chat.reset()
await chat.achat("message")
```

## IngestionPipeline

```python
from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[SentenceSplitter(chunk_size=512)],
    vector_store=vs,
)
nodes = pipeline.run(documents=docs, show_progress=True)
pipeline.persist("./pipeline_storage")
```

## Schema 类型

```python
from llama_index.core.schema import (
    Document,
    TextNode,
    NodeWithScore,
    QueryBundle,
    MetadataMode,
    MetadataFilter,
    MetadataFilters,
)

QueryBundle(query_str="...")
node.get_content(metadata_mode=MetadataMode.LLM)
```

## ResponseSynthesizer

```python
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode

syn = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    text_qa_template=PromptTemplate("...{context_str}...{query_str}..."),
)
```

## QueryEngine 组合

```python
from llama_index.core.query_engine import RetrieverQueryEngine

engine = RetrieverQueryEngine.from_args(
    retriever=index.as_retriever(similarity_top_k=10),
    node_postprocessors=[reranker],
    response_mode="compact",
)
```

## Agent

```python
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool

tools = [QueryEngineTool.from_defaults(query_engine=engine, description="KB")]
agent = ReActAgent.from_tools(tools, llm=Settings.llm)
result = await agent.run("用户问题")
```

## 常用集成 import

```python
# LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.ollama import Ollama
from llama_index.llms.llama_cpp import LlamaCPP

# Embedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# VectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

# Reader
from llama_index.readers.file import PDFReader

# Postprocessor
from llama_index.postprocessor.cohere_rerank import CohereRerank
```

## Callback 调试

```python
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

Settings.callback_manager = CallbackManager([LlamaDebugHandler(print_trace_on_end=True)])
```

## 核心 Base 类（扩展用）

| 基类 | 路径 |
|------|------|
| `BaseLLM` / `LLM` | `core/llms/llm.py` |
| `BaseEmbedding` | `core/base/embeddings/base.py` |
| `BasePydanticVectorStore` | `core/vector_stores/types.py` |
| `BaseRetriever` | `core/base/base_retriever.py` |
| `BaseQueryEngine` | `core/base/base_query_engine.py` |
| `BaseSynthesizer` | `core/response_synthesizers/base.py` |
| `BaseNodePostprocessor` | `core/postprocessor/types.py` |
| `TransformComponent` | `core/schema.py` |
| `BaseReader` | `core/readers/base.py` |

## 文件路径速查

| 功能 | 源码路径 |
|------|----------|
| 数据模型 | `core/schema.py` |
| 全局配置 | `core/settings.py` |
| 向量索引 | `core/indices/vector_store/base.py` |
| 查询引擎 | `core/query_engine/retriever_query_engine.py` |
| 摄取流水线 | `core/ingestion/pipeline.py` |
| 存储上下文 | `core/storage/storage_context.py` |
| Chat | `core/chat_engine/context.py` |
| ReAct Agent | `core/agent/workflow/react_agent.py` |
| OpenAILike LLM | `integrations/llms/llama-index-llms-openai-like/` |

## 环境变量（OpenAI 默认）

```bash
export OPENAI_API_KEY=sk-...
# 可选
export OPENAI_API_BASE=https://api.openai.com/v1
```

使用 OpenAILike 本地服务时 API Key 可设为占位符。

## 版本

本分析基于 **llama-index 0.14.23** / **llama-index-core ≥0.14.23, <0.15.0**。

API 在 0.10→0.14 间有 breaking changes（ServiceContext → Settings，Agent 迁移 Workflow），请以本仓库源码为准。
