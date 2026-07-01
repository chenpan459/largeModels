# 09 - Entrypoints 与 OpenAI API

## 入口概览

```
vllm/entrypoints/
├── openai/
│   ├── api_server.py      # FastAPI 主服务
│   ├── cli_args.py        # serve 子命令参数
│   ├── protocol.py        # Request/Response Pydantic 模型
│   ├── serving_chat.py    # /v1/chat/completions
│   ├── serving_completion.py
│   ├── serving_embedding.py
│   └── serving_*.py
├── launcher.py            # serve_http
└── llm.py                 # 高层 LLM class 导出
```

## 启动服务

```bash
# 推荐
vllm serve meta-llama/Llama-3.2-3B --host 0.0.0.0 --port 8000

# 等价
python -m vllm.entrypoints.openai.api_server --model ...
```

## api_server.py 架构

1. 解析 `AsyncEngineArgs` → 构建 `EngineClient`
2. 创建 `OpenAIServingChat` / `Completion` / `Embedding` 等
3. FastAPI routes 注册
4. `serve_http()` + uvloop

### EngineClient 类型

| Client | 场景 |
|--------|------|
| `AsyncLLMEngine` | 同进程 async |
| `MQLLMEngineClient` | 多进程 MQ 隔离 engine |

```python
from vllm.engine.protocol import EngineClient
```

## OpenAI 兼容端点

| 端点 | 处理器 |
|------|--------|
| `POST /v1/chat/completions` | `OpenAIServingChat` |
| `POST /v1/completions` | `OpenAIServingCompletion` |
| `POST /v1/embeddings` | `OpenAIServingEmbedding` |
| `POST /v1/rerank` | rerank 模型 |
| `POST /v1/tokenize` | tokenize |
| LoRA load/unload | adapter 管理 |

### Chat 流式

`StreamingResponse` + async generator，SSE `data: {...}` 格式，兼容 OpenAI SDK。

## Protocol 层

`protocol.py` — Pydantic 模型：

- `ChatCompletionRequest` / `Response`
- `CompletionRequest`
- `ErrorResponse`
- 扩展字段：top_k、repetition_penalty、structured outputs

## Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-3B", tensor_parallel_size=1)
params = SamplingParams(temperature=0.8, max_tokens=128)
outputs = llm.generate(["Hello"], params)
```

`LLM` 封装同步 `LLMEngine`，适合离线 batch。

## AsyncLLM

```python
from vllm.v1.engine.async_llm import AsyncLLM
```

供 FastAPI 高并发；与 `EngineCoreClient` 通信。

## 与 kefu-kb 集成

当前 kefu-kb 调用 **llama-server** OpenAI API。换 vLLM：

```yaml
# 等价配置
llama:
  base_url: http://127.0.0.1:8000/v1
  chat_model: meta-llama/Llama-3.2-3B
```

vLLM 优势：**高并发** + PagedAttention；llama-server 优势：**GGUF 本地**、依赖轻。

## 请求路径小结

```
HTTP → serving_chat.create_chat_completion
     → tokenizer + template
     → engine.generate(prompt_token_ids)
     → EngineCore (schedule loop)
     → detokenize stream
```

## 常用 serve 参数

| CLI | 含义 |
|-----|------|
| `--model` | HF model id 或路径 |
| `-tp` / `--tensor-parallel-size` | TP |
| `--max-model-len` | 最大上下文 |
| `--gpu-memory-utilization` | KV 显存比例 |
| `--enable-prefix-caching` | 前缀缓存 |
| `--dtype` | auto/half/bfloat16 |

## 安全与运维

- 生产需加 auth、rate limit（vLLM 基础版无内置）
- `--allowed-local-media-path` 等多模态路径限制
- 健康检查：通常 GET `/health` 或 `/v1/models`

## 调试

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve ...
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[{"role":"user","content":"hi"}]}'
```
