# 09 - Entrypoints 与 OpenAI API

## 入口概览

```
vllm/entrypoints/
├── llm.py                     # 公开 LLM 类（from vllm import LLM）
├── launcher.py                # serve_http()、uvicorn 启动
├── utils.py                   # 请求处理工具
└── openai/
    ├── api_server.py          # FastAPI 主应用（路由注册）
    ├── cli_args.py            # vllm serve 子命令参数
    ├── protocol.py            # Pydantic Request/Response
    ├── serving_engine.py      # OpenAIServing 基类
    ├── serving_chat.py        # /v1/chat/completions
    ├── serving_completion.py  # /v1/completions
    ├── serving_embedding.py   # /v1/embeddings
    ├── serving_pooling.py     # /pooling
    ├── serving_score.py       # /score、/v1/score
    ├── serving_tokenization.py
    ├── serving_transcription.py
    ├── tool_parsers.py        # function calling 解析
    ├── reasoning_parsers.py   # reasoning model 解析
    └── logits_processors.py   # OpenAI 自定义 logits（V0）
```

## 启动服务

```bash
# 推荐方式
vllm serve meta-llama/Llama-3.2-3B --host 0.0.0.0 --port 8000

# 等价
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B --host 0.0.0.0 --port 8000
```

## api_server.py 架构

### 1. 构建 Engine Client

`build_async_engine_client_from_engine_args()`（`api_server.py:168-243`）：

| 条件 | Client |
|------|--------|
| `VLLM_USE_V1=True`（默认） | `v1.engine.async_llm.AsyncLLM`（**始终子进程**） |
| V0 + MQ 可用 | `MQLLMEngineClient`（spawn engine 进程，ZMQ IPC） |
| V0 + `--disable-frontend-multiprocessing` | 同进程 `AsyncLLMEngine` |

V1 忽略 `--disable-frontend-multiprocessing`（打 warning）。

### 2. 创建 Serving 实例

```python
openai_serving_chat = OpenAIServingChat(engine_client, model_config, ...)
openai_serving_completion = OpenAIServingCompletion(...)
openai_serving_embedding = OpenAIServingEmbedding(...)
# ...
```

### 3. 注册 FastAPI Routes

### 4. `serve_http()` + uvloop 启动

## HTTP 路由完整表

| 方法 | 路径 | 处理器 | 说明 |
|------|------|--------|------|
| GET | `/health` | health check | 健康检查 |
| GET | `/ping` | 同 health | SageMaker 兼容 |
| GET | `/load` | 负载指标 | 当前负载 |
| GET | `/v1/models` | 模型列表 | OpenAI 兼容 |
| GET | `/version` | 版本信息 | vLLM version |
| POST | `/v1/chat/completions` | `OpenAIServingChat` | Chat（主力） |
| POST | `/v1/completions` | `OpenAIServingCompletion` | 补全 |
| POST | `/v1/embeddings` | `OpenAIServingEmbedding` | Embedding（V0 pooling） |
| POST | `/pooling` | `OpenAIServingPooling` | Pooling 模型 |
| POST | `/score` | `OpenAIServingScore` | 打分 |
| POST | `/v1/score` | 同上 | |
| POST | `/v1/rerank` | rerank | 重排序 |
| POST | `/rerank` | 同上 | |
| POST | `/v2/rerank` | 同上 | Cohere 兼容 |
| POST | `/v1/audio/transcriptions` | transcription | 语音 |
| POST | `/tokenize` | tokenization | 分词 |
| POST | `/detokenize` | detokenization | 反分词 |
| POST | `/reset_prefix_cache` | engine | 清 prefix cache（dev） |
| POST | `/sleep` | power mgmt | 显存 offload |
| POST | `/wake_up` | power mgmt | 恢复 |
| GET | `/is_sleeping` | power mgmt | 状态 |
| POST | `/start_profile` | profiling | torch profiler |
| POST | `/stop_profile` | profiling | |
| POST | `/v1/load_lora_adapter` | LoRA | 热加载 adapter |
| POST | `/v1/unload_lora_adapter` | LoRA | 卸载 |
| POST | `/invocations` | SageMaker | AWS 兼容 |

## EngineClient 协议

`vllm/engine/protocol.py` — 抽象接口：

```python
class EngineClient(Protocol):
    async def generate(...) -> AsyncIterator[RequestOutput]: ...
    async def abort(request_id: str) -> None: ...
    async def get_model_config() -> ModelConfig: ...
    async def reset_prefix_cache() -> None: ...
    # ...
```

实现类：

| 类 | 场景 |
|----|------|
| `AsyncLLM` | V1 默认 |
| `AsyncLLMEngine` | V0 同进程 |
| `MQLLMEngineClient` | V0 多进程 MQ |

## Chat 请求路径

```
POST /v1/chat/completions
  → OpenAIServingChat.create_chat_completion()
  → 解析 ChatCompletionRequest（protocol.py）
  → apply_chat_template()（tokenizer）
  → 构建 SamplingParams
  → engine_client.generate(prompt_token_ids, sampling_params, request_id)
      → AsyncLLM.add_request()
      → Processor → EngineCoreClient → EngineCore
  → async for output in generator:
      → delta text → SSE chunk
  → StreamingResponse（text/event-stream）
```

非流式：等待 generator 完成，返回完整 `ChatCompletionResponse`。

### 流式格式

```
data: {"id":"...","choices":[{"delta":{"content":"Hello"}}],...}

data: [DONE]
```

兼容 OpenAI Python SDK。

## OpenAIServing 基类

`serving_engine.py` → `OpenAIServing`：

- 公共错误处理（`ErrorResponse`）
- Model name 路由（`OpenAIServingModels`）
- Tokenizer 访问
- Request ID 生成

子类实现具体 endpoint 逻辑。

## Protocol 层

`protocol.py` — Pydantic 模型：

| 模型 | 用途 |
|------|------|
| `ChatCompletionRequest` | chat 输入 |
| `ChatCompletionResponse` | chat 输出 |
| `CompletionRequest` | 补全输入 |
| `EmbeddingRequest` | embedding 输入 |
| `ErrorResponse` | 错误格式 |

vLLM 扩展字段（非 OpenAI 标准）：

- `top_k`、`repetition_penalty`、`min_p`
- `guided_json`、`guided_regex`（structured output）
- `logits_processors`（V0）
- `priority`（V0 scheduling）

## Python 同步 API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-3B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)
params = SamplingParams(temperature=0.8, max_tokens=128, top_p=0.95)
outputs = llm.generate(["Hello, my name is"], params)
for o in outputs:
    print(o.outputs[0].text)
```

`LLM`（`entrypoints/llm.py`）封装：

- `LLMEngine`（V1 别名）
- `EngineArgs` 解析
- batch generate、beam search（V0）

## AsyncLLM

```python
from vllm.v1.engine.async_llm import AsyncLLM

engine = AsyncLLM.from_engine_args(engine_args)
async for output in engine.generate(...):
    print(output)
```

- 供 FastAPI 高并发
- 始终 AsyncMPClient + EngineCoreProc
- 实现 `EngineClient` protocol

## Tool Calling 与 Reasoning

- `tool_parsers.py` — 各模型 tool call 格式解析（Llama、Qwen、DeepSeek 等）
- `reasoning_parsers.py` — reasoning model（如 DeepSeek R1）thinking 段解析

## 与 kefu-kb 集成

当前 kefu-kb 使用 **llama-server** OpenAI API。换 vLLM：

```yaml
# config.yaml
llama:
  base_url: http://127.0.0.1:8000/v1
  chat_model: meta-llama/Llama-3.2-3B  # 与 vllm serve --model 一致
```

```bash
# 启动 vLLM
vllm serve meta-llama/Llama-3.2-3B --host 0.0.0.0 --port 8000
```

| | vLLM | llama-server |
|---|------|--------------|
| 权重格式 | HuggingFace | GGUF |
| 并发 | PagedAttention + continuous batch | 中等 |
| 依赖 | GPU + 较重 | 轻量 CPU/GPU |
| Embedding | 需 pooling 模型 | `/v1/embeddings` |

kefu-kb 当前用本地 sentence-transformers embedding，只需 vLLM 提供 **chat** 即可。

## 常用 serve 参数

| CLI | 含义 |
|-----|------|
| `--model` | HF model id 或本地路径 |
| `-tp` / `--tensor-parallel-size` | TP 度 |
| `--max-model-len` | 最大上下文 |
| `--gpu-memory-utilization` | KV 显存比例 |
| `--enable-prefix-caching` | 前缀缓存（V1 默认开） |
| `--dtype` | auto/float16/bfloat16 |
| `--quantization` | awq/gptq/fp8/... |
| `--max-num-seqs` | 最大并发 |
| `--max-num-batched-tokens` | 单 step token 上限 |
| `--enforce-eager` | 禁用 CUDA graph |
| `--distributed-executor-backend` | mp / ray |

## 安全与运维

vLLM 基础版 **无内置** auth / rate limit。生产建议：

- 前置 nginx/envoy + API key
- `--allowed-local-media-path` 限制多模态本地文件
- 健康检查：`GET /health` 或 `/v1/models`
- Prometheus metrics（若启用 observability config）

Power management（dev/运维）：

```bash
curl -X POST http://localhost:8000/sleep
curl -X POST http://localhost:8000/wake_up
```

## 调试

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-3B --port 8000

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": true
  }'
```

## 关键源码

| 主题 | 文件 |
|------|------|
| Client 选择 | `api_server.py:168-243` |
| Chat serving | `serving_chat.py` |
| LLM class | `entrypoints/llm.py` |
| Protocol | `protocol.py` |
| EngineClient | `engine/protocol.py` |
