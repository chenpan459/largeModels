# 04 - Settings 全局配置

源码：`llama-index-core/llama_index/core/settings.py`

## 概述

`Settings` 是 `@dataclass` 包装的单例 `_Settings`，通过模块级代理暴露：

```python
from llama_index.core import Settings

Settings.llm = ...
Settings.embed_model = ...
```

所有未显式传入 LLM/Embedding 的组件（Index、QueryEngine、Synthesizer）都会回退到 `Settings`。

## 可配置项

| 属性 | 类型 | 默认值行为 |
|------|------|------------|
| `llm` | `LLM` | `resolve_llm("default")` → 通常 OpenAI |
| `embed_model` | `BaseEmbedding` | `resolve_embed_model("default")` |
| `node_parser` | `NodeParser` | `SentenceSplitter()` |
| `callback_manager` | `CallbackManager` | 空 CallbackManager |
| `tokenizer` | `Callable` | 按 LLM 解析 |
| `prompt_helper` | `PromptHelper` | 按 context_window 计算 |
| `transformations` | `List[TransformComponent]` | 默认含 node_parser |

## 典型配置示例

### OpenAI

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
```

### 本地 llama-server（OpenAI 兼容）

```python
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

Settings.llm = OpenAILike(
    model="Qwen2.5-7B-Instruct",
    api_base="http://127.0.0.1:8080/v1",
    api_key="not-needed",
    is_chat_model=True,
    context_window=8192,
)
Settings.embed_model = OpenAILikeEmbedding(
    model_name="bge-m3",
    api_base="http://127.0.0.1:8081/v1",
    api_key="not-needed",
)
```

### 切分参数

```python
from llama_index.core.node_parser import SentenceSplitter

Settings.node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=64,
)
# 等价于设置 transformations
Settings.transformations = [Settings.node_parser]
```

## resolve 机制

`embeddings/utils.py` 与 `llms/utils.py` 中的 `resolve_*` 函数支持：

- 直接传入实例
- 传入字符串 shorthand（`"default"`, `"local"`, 模型名）
- 从环境变量推断

这使得 Notebook 中 `VectorStoreIndex.from_documents(docs)` 无需每次传 embed_model。

## PromptHelper

根据 LLM `context_window` 计算：

- 检索 chunk 如何塞进 prompt
- `num_output` 预留 token

```python
Settings.prompt_helper  # 自动绑定 Settings.llm
```

Chat 模型使用 `ChatPromptHelper`（`is_chat_model()` 为真时）。

## CallbackManager

```python
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

debug_handler = LlamaDebugHandler(print_trace_on_end=True)
Settings.callback_manager = CallbackManager([debug_handler])
```

事件类型见 `callbacks/schema.py`：`CHUNKING`, `EMBEDDING`, `RETRIEVE`, `LLM`, `SYNTHESIZE` 等。

## 与 ServiceContext 的关系

旧版 API 使用 `ServiceContext`（`service_context.py`），现已 **deprecated**，统一迁移到 `Settings`：

```python
# 旧（勿用）
# from llama_index.core import ServiceContext

# 新
from llama_index.core import Settings
```

## 多租户 / 多配置

`Settings` 是全局单例，**不适合** 同一进程内多租户不同 LLM。可选方案：

1. 每个请求显式传入 `llm` / `embed_model` 到 Index / QueryEngine
2. 使用 `contextvars` 封装（社区模式）
3. 每租户独立 worker 进程

## kefu-kb 映射

kefu-kb 的 `config.yaml` 等价于 LlamaIndex Settings 的分字段配置：

| kefu-kb | LlamaIndex |
|---------|------------|
| `llama.base_url` | `OpenAILike.api_base` |
| `llama.chat_model` | `OpenAILike.model` |
| `llama.embed_model` | `OpenAILikeEmbedding.model_name` |
| `chunk_size` / `overlap` | `SentenceSplitter` 参数 |
