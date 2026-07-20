# 13 - llama-server 的 OpenAILike 集成

> 基于 `llama-index` 0.14.23。本章只给出一条规范路径：**llama.cpp 的
> `llama-server` 提供 OpenAI-compatible HTTP API，LlamaIndex 通过
> `OpenAILike`/`OpenAILikeEmbedding` 访问**。这与进程内
> `llama_index.llms.llama_cpp.LlamaCPP` 是两种部署拓扑，不应混用配置。

## 1. 组件边界

```text
documents
  -> SentenceSplitter
  -> OpenAILikeEmbedding
  -> llama-server :8081 /v1/embeddings
  -> VectorStoreIndex

query
  -> retriever
  -> response synthesizer
  -> OpenAILike
  -> llama-server :8080 /v1/chat/completions
```

LlamaIndex 不读取 GGUF，也不管理 llama.cpp 推理进程。它只看到 HTTP 协议。模型路径、
GPU offload、并发槽位、KV cache 和 chat template 均由 `llama-server` 负责。

对应源码：

- LLM：
  `llama-index-integrations/llms/llama-index-llms-openai-like/llama_index/llms/openai_like/base.py`
- Embedding：
  `llama-index-integrations/embeddings/llama-index-embeddings-openai-like/llama_index/embeddings/openai_like/base.py`
- 上游 OpenAI 客户端：
  `llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py`

## 2. 安装

```bash
pip install \
  "llama-index-core>=0.14.23,<0.15" \
  "llama-index-llms-openai-like>=0.7,<0.8" \
  llama-index-embeddings-openai-like
```

若接外部向量库，再单独安装对应插件；最小示例使用 core 的
`SimpleVectorStore`，避免把服务对接与数据库配置混在一起。

## 3. 启动两个 llama-server

下面的 flags 需以本机 llama.cpp build 的 `llama-server --help` 为准；不同 llama.cpp
提交可能改名。Chat 模型和 embedding 模型应分进程部署：

```bash
# 终端 1：生成模型
/path/to/llama-server \
  -m /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 8192

# 终端 2：向量模型
/path/to/llama-server \
  -m /models/bge-m3.gguf \
  --host 127.0.0.1 \
  --port 8081 \
  --embedding
```

先用服务自身的健康端点/日志确认模型加载，再确认 OpenAI-compatible 路径：

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8081/v1/models
```

若模型 ID 与示例不同，应使用服务实际返回/接受的 ID。不要假设 GGUF 文件名一定就是
API model 名。

## 4. 唯一规范 Python 配置

```python
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

llm = OpenAILike(
    model="Qwen2.5-7B-Instruct",
    api_base="http://127.0.0.1:8080/v1",
    api_key="local-placeholder",
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=8192,
    max_tokens=768,
    temperature=0.1,
    timeout=120.0,
)

embed_model = OpenAILikeEmbedding(
    model_name="bge-m3",
    api_base="http://127.0.0.1:8081/v1",
    api_key="local-placeholder",
    embed_batch_size=8,
    timeout=120.0,
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
```

为什么这些字段不能省略：

- `api_base` 必须包含服务实际支持的 `/v1` 前缀。
- OpenAI 客户端要求 key；本地服务不校验时仍给一个非空占位值。
- `is_chat_model=True` 让 `chat()` 走 chat-completions；否则消息会先拼成 completion
  prompt。
- `is_function_calling_model=False` 防止 Agent 自动选用服务未必支持的原生 tools。
- `context_window` 是 LlamaIndex 的预算元数据，应与 server 的 `-c` 协调。
- Embedding 构造参数是 `model_name`，不是 `model`。

## 5. 建索引、持久化和查询

```python
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

documents = SimpleDirectoryReader("./data", recursive=True).load_data()
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    show_progress=True,
)
index.set_index_id("local-kb")
index.storage_context.persist("./storage/local-kb")

storage = StorageContext.from_defaults(persist_dir="./storage/local-kb")
index = load_index_from_storage(
    storage,
    index_id="local-kb",
    embed_model=embed_model,
)

engine = index.as_query_engine(llm=llm, similarity_top_k=5)
response = await engine.aquery("退货政策是什么？")
print(response.response)
for item in response.source_nodes:
    print(item.score, item.node.metadata)
```

持久化的 simple vector store 已包含 embedding，不会在加载时重新向 :8081 请求；查询
仍需 embedding 当前问题。因此换 embedding 模型或维数后必须重建索引。

## 6. 调用链与同步/异步/流式

### Embedding

`VectorStoreIndex.from_documents()` 经 transformations 产生 nodes，再调用
`OpenAILikeEmbedding`。它继承 `OpenAIEmbedding`，后者使用 OpenAI SDK 请求
`:8081/v1/embeddings`；批大小来自 `embed_batch_size`。

### 非流式查询

```text
await query_engine.aquery(query)
  -> retriever.aretrieve()
  -> embed_model.aget_query_embedding()
  -> vector_store.aquery()/query()
  -> response_synthesizer.asynthesize()
  -> llm.achat()
  -> OpenAI async client
  -> POST :8080/v1/chat/completions
```

向量 store 是否真正异步取决于其插件实现；simple store 主要是内存计算。

### 流式查询

```python
engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=5,
    streaming=True,
)
response = await engine.aquery("解释退款流程")
async for delta in response.async_response_gen():
    print(delta, end="", flush=True)
```

检索和 prompt 构造完成后，`OpenAILike.astream_chat()` 才请求 streaming chat
completion。流式减少生成阶段的可见等待，不会流式化 embedding 或向量检索。同步代码
则使用 `query()` 和 `response.response_gen`。

## 7. ChatEngine

```python
chat = index.as_chat_engine(
    chat_mode="condense_plus_context",
    llm=llm,
    similarity_top_k=5,
    system_prompt="只根据检索资料回答；资料不足时说明不知道。",
)

stream = await chat.astream_chat("那退款运费呢？")
async for delta in stream.async_response_gen():
    print(delta, end="")
```

有历史时该模式可能先调用一次 `llm.acomplete()` 改写追问，再检索，再调用 chat
completion 生成答案，所以一次用户请求不一定只对应一次 :8080 请求。

## 8. Agent 能力边界

普通 llama-server chat-completions 并不自动等于可靠的 OpenAI tools 实现。默认保持
`is_function_calling_model=False`，需要 Agent 时使用：

```python
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent

agent = ReActAgent(tools=[kb_tool], llm=llm)
workflow = AgentWorkflow(agents=[agent])
result = await workflow.run(user_msg="查询政策后给出办理步骤")
```

只有在当前 llama.cpp 版本、模型 chat template 和实测响应都正确支持 tool calls 时，
才应改 flag 并使用 `FunctionAgent`。

## 9. 故障定位

- 404：通常是 `api_base` 前缀或 server 版本不匹配。
- 401：服务启用鉴权时占位 key 无效，需传真实 token。
- 输出像 prompt 回显：检查 `is_chat_model`、模型 chat template 和 instruct 模型类型。
- context overflow：同时核对 server `-c`、`context_window`、chunk/top-k、历史和
  `max_tokens`；客户端元数据不会扩大服务端 KV cache。
- embedding 维数错误：旧 collection/索引与新模型不一致，必须新建或重建。
- 首 token 慢：可能耗时在追问改写、embedding、检索或 prompt eval，而非 streaming
  失效。
- 并发超时：调大客户端 timeout 不能增加 server slots；应同时调整 llama-server
  并发和应用限流。
- 不要把 `OpenAILike` 写成“只改一个 URL”：能力 flags、token 预算、异步客户端和
  服务协议子集都会影响上层行为。
