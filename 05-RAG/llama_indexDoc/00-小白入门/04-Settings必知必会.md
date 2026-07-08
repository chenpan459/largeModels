# 04 - Settings 必知必会

## Settings 是什么？

**整个 Python 进程里的一份「全局配置单」**，告诉 LlamaIndex：

- 用哪个 **LLM** 生成答案
- 用哪个 **Embedding** 把文字变向量
- 用哪种方式 **切分** 文档

```python
from llama_index.core import Settings

Settings.llm = ...
Settings.embed_model = ...
Settings.node_parser = ...
```

## 为什么必须手动配置？（本地部署）

如果你不配置，`from_documents()` 内部会用默认的 `"default"` 解析器 → **尝试 OpenAI** → 报错：

```
OpenAIError: The OPENAI_API_KEY environment variable is not set
```

源码逻辑（简化）：`Settings.embed_model` 为 `None` 时 → `resolve_embed_model("default")` → 需要 `OPENAI_API_KEY`。

**结论：用 llama-server 时，第一行有效代码就应该是 Settings。**

## 本地 llama-server 标准写法

```python
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

Settings.llm = OpenAILike(
    model="你的对话模型名",
    api_base="http://127.0.0.1:8080/v1",   # 必须带 /v1
    api_key="not-needed",                  # 不能为空字符串时部分客户端会报错
    is_chat_model=True,                      # ⚠️ 默认 False，llama-server 必须 True
    context_window=8192,                     # 与 GGUF n_ctx 接近
    temperature=0.1,
)

Settings.embed_model = OpenAILikeEmbedding(
    model_name="你的embedding模型名",        # ⚠️ 参数名是 model_name 不是 model
    api_base="http://127.0.0.1:8081/v1",
    api_key="not-needed",
    embed_batch_size=8,                      # 批量 embedding，减轻 server 压力
)

Settings.node_parser = SentenceSplitter(
    chunk_size=256,      # 单位是 token，中文建议 256~512
    chunk_overlap=32,
)
```

## 参数易错表

| 参数 | 错误写法 | 正确写法 |
|------|----------|----------|
| 对话 API | `is_chat_model=False`（默认） | `is_chat_model=True` |
| Embedding 模型名 | `model="bge-m3"` | `model_name="bge-m3"` |
| API 地址 | `http://127.0.0.1:8080` | `http://127.0.0.1:8080/v1` |
| API Key | 留空 `""` | `"not-needed"` 或 `"fake"` |

## OpenAI 用户

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# 可以不写 Settings，用默认 OpenAI
# 或显式：
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
```

安装：`pip install llama-index`（元包已含 OpenAI 集成）。

## ServiceContext 已废弃

网上很多旧教程写：

```python
from llama_index.core import ServiceContext  # ❌ 0.10+ 请用 Settings
```

**一律改用 `Settings`**。详见 [08-常见报错与修复.md](./08-常见报错与修复.md)。

## Settings 是单例的利弊

| 优点 | 缺点 |
|------|------|
| 代码简单，不用到处传参 | 一个进程只能一套默认 LLM/Embedding |
| 与 `from_documents` 无缝配合 | 多租户 API 要在每个组件上单独 override |

局部覆盖示例：

```python
engine = index.as_query_engine(llm=另一个OpenAILike(...))
```

## 改 Settings 后要不要重建索引？

| 改了什么 | 要不要重建 |
|----------|-----------|
| 只改 `llm` | **不要**（只影响生成） |
| 改了 `embed_model` | **要**（向量维度/语义空间变了） |
| 改了 `node_parser` chunk 大小 | **要**（chunk 边界变了） |

## 无服务器时练手：MockEmbedding

```python
from llama_index.core.embeddings import MockEmbedding
Settings.embed_model = MockEmbedding(embed_dim=384)
```

仅用于测通代码流程，**不能**用于真实问答质量评估。

## 下一步

→ [05-读文档与切分.md](./05-读文档与切分.md)  
→ 进阶：[04-settings-config.md](../04-settings-config.md)
