# 14 - LlamaIndex 0.14.23 API 与源码路径速查

> 本页只列本 checkout 可验证的当前 API。插件必须另行安装；导入成功不代表远端服务实现
> 了所有可选协议能力。

## 1. Core 顶层

```python
from llama_index.core import (
    Document,
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
    load_indices_from_storage,
)
```

主要 re-export 定义于
`llama-index-core/llama_index/core/__init__.py`。

```python
Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = splitter
Settings.transformations = [splitter]
Settings.callback_manager = callback_manager
```

源码：`llama_index/core/settings.py`。

## 2. Schema 与 metadata filter

节点/消息数据类型：

```python
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
```

向量过滤类型不在 `core.schema`，正确导入为：

```python
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="tenant_id",
            value="tenant-a",
            operator=FilterOperator.EQ,
        )
    ],
    condition=FilterCondition.AND,
)
retriever = index.as_retriever(similarity_top_k=5, filters=filters)
```

源码：`llama_index/core/vector_stores/types.py`。具体 VectorStore 是否支持某个 operator
需查插件实现。

## 3. Node parsing 与 ingestion

```python
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionCache,
    IngestionPipeline,
)
from llama_index.core.node_parser import SentenceSplitter

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=64),
        embed_model,
    ],
    vector_store=vector_store,
)
nodes = pipeline.run(documents=documents, show_progress=True)
nodes = await pipeline.arun(documents=documents)
pipeline.persist("./pipeline-cache")
```

`IngestionPipeline.persist()` 保存 pipeline/cache 状态，不等于
`StorageContext.persist()` 的 index/doc/vector 快照。

源码：`llama_index/core/ingestion/pipeline.py`。

## 4. Index 创建、更新与恢复

```python
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    storage_context=storage_context,
)
index = VectorStoreIndex(nodes, embed_model=embed_model)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

index.insert(document)
index.insert_nodes(nodes)
index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
index.set_index_id("kb-v1")

index.storage_context.persist("./storage")
storage = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage, index_id="kb-v1")
```

源码：

- 基类与 `as_query_engine/as_chat_engine`：`core/indices/base.py`
- 向量索引：`core/indices/vector_store/base.py`
- 加载：`core/indices/loading.py`

## 5. Retriever 与 QueryEngine

```python
retriever = index.as_retriever(similarity_top_k=5, filters=filters)
nodes = retriever.retrieve("问题")
nodes = await retriever.aretrieve("问题")

engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=5,
    response_mode="compact",
)
response = engine.query("问题")
response = await engine.aquery("问题")
print(response.response)
print(response.source_nodes)
```

显式组合：

```python
from llama_index.core.query_engine import RetrieverQueryEngine

engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=llm,
    node_postprocessors=[reranker],
    response_mode="compact",
)
```

流式：

```python
engine = index.as_query_engine(llm=llm, streaming=True)

sync_response = engine.query("问题")
for delta in sync_response.response_gen:
    print(delta, end="")

async_response = await engine.aquery("问题")
async for delta in async_response.async_response_gen():
    print(delta, end="")
```

源码：`core/query_engine/retriever_query_engine.py` 及
`core/response_synthesizers/`。

## 6. ChatEngine

```python
from llama_index.core.chat_engine import (
    AgentChatResponse,
    BaseChatEngine,
    ChatMode,
    CondensePlusContextChatEngine,
    ContextChatEngine,
    SimpleChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.core.memory import Memory

memory = Memory.from_defaults(token_limit=6000)
chat = index.as_chat_engine(
    chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
    llm=llm,
    memory=memory,
)

response = chat.chat("问题")
response = await chat.achat("问题")
chat.reset()
history = chat.chat_history
```

可用模式为 `SIMPLE`、`CONDENSE_QUESTION`、`CONTEXT`、
`CONDENSE_PLUS_CONTEXT`、`BEST`。枚举中的 `REACT` 和 `OPENAI` 已移除实现，传入
`as_chat_engine()` 会抛 `ValueError`。

流式：

```python
stream = chat.stream_chat("问题")
for delta in stream.response_gen:
    ...

stream = await chat.astream_chat("问题")
async for delta in stream.async_response_gen():
    ...
```

源码：`core/chat_engine/types.py` 和各具体 engine 文件。

## 7. Memory 与 chat store

推荐：

```python
from llama_index.core.memory import BaseMemory, Memory

memory = Memory.from_defaults(token_limit=6000)
await memory.aput(message)
messages = await memory.aget(input="当前问题")
```

兼容旧代码但已 deprecated：

```python
from llama_index.core.memory import ChatMemoryBuffer
```

Chat store：

```python
from llama_index.core.storage.chat_store import (
    BaseChatStore,
    SimpleChatStore,
)

store = SimpleChatStore()
store.add_message("session-1", message)
store.persist("./chat_store.json")
store = SimpleChatStore.from_persist_path("./chat_store.json")
```

源码：`core/memory/` 与 `core/storage/chat_store/`。

## 8. Storage

```python
from llama_index.core.storage.docstore import (
    BaseDocumentStore,
    SimpleDocumentStore,
)
from llama_index.core.storage.index_store import (
    BaseIndexStore,
    SimpleIndexStore,
)
from llama_index.core.storage.storage_context import StorageContext

storage = StorageContext.from_defaults(
    docstore=docstore,
    index_store=index_store,
    vector_store=vector_store,       # 写入 default namespace
)
storage.add_vector_store(other_store, namespace="archive")
storage.persist("./storage")
```

默认向量 namespace 是 `default`，分隔符为 `__`，另有 `image` namespace。源码：
`core/storage/storage_context.py`、`core/vector_stores/simple.py`。

## 9. Tools

```python
from llama_index.core.tools import (
    AsyncBaseTool,
    BaseTool,
    FunctionTool,
    QueryEngineTool,
    RetrieverTool,
    ToolOutput,
)

fn_tool = FunctionTool.from_defaults(
    fn=lookup_order,
    name="lookup_order",
    description="按订单号查询状态",
)
kb_tool = QueryEngineTool.from_defaults(
    query_engine=engine,
    name="knowledge_base",
    description="查询政策知识库",
)
```

源码：`core/tools/`。

## 10. 当前 Workflow Agent

```python
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    CodeActAgent,
    FunctionAgent,
    ReActAgent,
)

agent = ReActAgent(
    tools=[kb_tool, fn_tool],
    llm=llm,
    system_prompt="使用工具核实事实。",
)
workflow = AgentWorkflow(agents=[agent])
result = await workflow.run(user_msg="查询订单并解释政策")
```

当前构造是 `ReActAgent(tools=..., llm=...)`，没有
`ReActAgent.from_tools()`；执行循环由 `AgentWorkflow` 负责，不应使用旧
`AgentRunner` 说法。

Function calling：

```python
agent = FunctionAgent(
    tools=[fn_tool],
    llm=function_calling_llm,
    allow_parallel_tool_calls=True,
)
workflow = AgentWorkflow(agents=[agent])
```

`FunctionAgent` 要求 `llm.metadata.is_function_calling_model=True`。

事件流：

```python
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentStream,
    ToolCall,
    ToolCallResult,
)

handler = workflow.run(user_msg="执行任务")
async for event in handler.stream_events():
    if isinstance(event, AgentStream):
        print(event.delta, end="")
result: AgentOutput = await handler
```

源码：`core/agent/workflow/multi_agent_workflow.py`、
`react_agent.py`、`function_agent.py`、`codeact_agent.py`、
`workflow_events.py`。

## 11. 自定义 Workflow

```python
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

class Retrieved(Event):
    nodes: list
    query: str

class RAGWorkflow(Workflow):
    @step
    async def retrieve(self, ev: StartEvent) -> Retrieved:
        query = ev.query
        return Retrieved(
            nodes=await retriever.aretrieve(query),
            query=query,
        )

    @step
    async def synthesize(self, ev: Retrieved) -> StopEvent:
        result = await synthesizer.asynthesize(ev.query, ev.nodes)
        return StopEvent(result=result)

result = await RAGWorkflow(timeout=60).run(query="问题")
```

`llama_index.core.workflow` 主要 re-export `workflows` 依赖中的实现。步骤以 Event 类型
路由，不是按源码书写顺序顺序调用。

## 12. Prompt 与 response synthesizer

```python
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)

template = PromptTemplate(
    "资料：\n{context_str}\n\n问题：{query_str}\n回答："
)
synthesizer = get_response_synthesizer(
    llm=llm,
    response_mode=ResponseMode.COMPACT,
    text_qa_template=template,
)
```

模板变量必须与具体 synthesizer 预期一致。

## 13. 常用插件导入

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

# Vector stores
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

# Readers / postprocessors
from llama_index.readers.file import PDFReader
from llama_index.postprocessor.cohere_rerank import CohereRerank
```

每一行都要求安装对应 `llama-index-{category}-{name}` 发行包。

## 14. OpenAI-compatible 本地服务

```python
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

llm = OpenAILike(
    model="local-chat",
    api_base="http://127.0.0.1:8080/v1",
    api_key="local-placeholder",
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=8192,
)
embed_model = OpenAILikeEmbedding(
    model_name="local-embedding",
    api_base="http://127.0.0.1:8081/v1",
    api_key="local-placeholder",
)
```

源码路径见 [13-llama.cpp本地推理集成.md](./13-llama.cpp本地推理集成.md)。

## 15. 易错 API 对照

| 过期/错误写法 | 0.14.23 写法 |
|---|---|
| `ReActAgent.from_tools(tools, llm=...)` | `ReActAgent(tools=tools, llm=...)` |
| 用 `AgentRunner` 描述当前执行器 | `AgentWorkflow(agents=[agent])` |
| `chat_mode="react"` / `"openai"` | 显式 `ReActAgent` / `FunctionAgent` |
| `ChatMemoryBuffer` 作为首选 | `Memory.from_defaults(...)` |
| `MetadataFilters` 从 `core.schema` 导入 | 从 `core.vector_stores.types` 导入 |
| `OpenAILikeEmbedding(model=...)` | `OpenAILikeEmbedding(model_name=...)` |
| `persist_dir="s3://..."` 自动识别 | 显式创建并传入 `fs=` |

## 16. 版本边界

本分析针对：

```text
llama-index 0.14.23
llama-index-core >=0.14.23,<0.15
```

集成包独立发版，版本号不与 core 相同。升级时应同时检查 core 与每个插件的
`pyproject.toml` 依赖范围，并优先以当前安装版本的源码签名为准。
