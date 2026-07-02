# 08 — 推理与 Chat

## 模块概览

推理相关代码分布在 `chat/` 和 `api/` 两个模块：

```
chat/chat_model.py     ← 统一推理门面
    ├── hf_engine.py       HuggingFace 后端
    ├── vllm_engine.py     vLLM 后端
    └── sglang_engine.py     SGLang 后端

api/app.py             ← FastAPI OpenAI 兼容服务
    ├── chat.py            对话处理
    └── protocol.py        请求/响应模型
```

## ChatModel 门面

`chat/chat_model.py` 的 `ChatModel` 是推理的统一入口，被 API、CLI chat 和 Web UI 共用。

### 初始化

```python
class ChatModel:
    def __init__(self, args=None):
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        # 按 infer_backend 选择引擎
        if model_args.infer_backend == EngineName.HF:
            self.engine = HuggingfaceEngine(...)
        elif model_args.infer_backend == EngineName.VLLM:
            self.engine = VllmEngine(...)
        elif model_args.infer_backend == EngineName.SGLANG:
            self.engine = SGLangEngine(...)
        # 后台 asyncio 事件循环（支持同步 + 异步调用）
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, ...)
```

### 接口方法

| 方法 | 类型 | 说明 |
|------|------|------|
| `chat()` | 同步 | 单次对话，返回完整回复 |
| `stream_chat()` | 同步 | 流式对话，逐 token 生成 |
| `get_scores()` | 同步 | 奖励模型打分 |
| `achat()` | 异步 | 异步单次对话 |
| `astream_chat()` | 异步 | 异步流式对话 |
| `aget_scores()` | 异步 | 异步打分 |

## 推理引擎

### HuggingFace Engine（默认）

`chat/hf_engine.py` — 基于 Transformers 的原生推理：

- 支持 LoRA adapter 热加载
- 支持多模态（图像/视频/音频）
- 兼容所有模型
- 适合开发调试和小批量推理

### vLLM Engine

`chat/vllm_engine.py` — 高吞吐推理：

- PagedAttention 内存管理
- 连续 batching
- 适合生产部署和高并发场景
- 需安装：`pip install -r requirements/vllm.txt`

### SGLang Engine

`chat/sglang_engine.py` — 结构化生成：

- RadixAttention 前缀缓存
- 适合复杂生成任务
- 需安装：`pip install -r requirements/sglang.txt`

### 引擎选择

在推理 YAML 或 CLI 中配置：

```yaml
infer_backend: huggingface  # huggingface / vllm / sglang
```

## FastAPI 服务

### 启动

```bash
llamafactory-cli api
# 或指定推理配置
API_MODEL_NAME=my-model llamafactory-cli api
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_HOST` | `0.0.0.0` | 监听地址 |
| `API_PORT` | `8000` | 监听端口 |
| `API_KEY` | — | API 密钥（可选） |
| `API_MODEL_NAME` | `gpt-3.5-turbo` | 模型名称 |
| `FASTAPI_ROOT_PATH` | — | 反向代理路径前缀 |

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/models` | 列出可用模型 |
| `POST` | `/v1/chat/completions` | 对话补全（流式/非流式） |
| `POST` | `/v1/score/evaluation` | 奖励模型打分 |

### 对话请求示例

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "stream": false
  }'
```

### 流式响应

设置 `"stream": true` 时，返回 SSE（Server-Sent Events）格式：

```
data: {"id":"...","choices":[{"delta":{"content":"你"},"index":0}]}

data: {"id":"...","choices":[{"delta":{"content":"好"},"index":0}]}

data: [DONE]
```

### 多模态请求

API 支持图像输入（base64 或 URL）：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ]
}
```

### 奖励模型打分

`POST /v1/score/evaluation` 用于 RM 推理，返回每个回复的分数：

```json
{
  "model": "reward-model",
  "messages": [
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "回答1"},
    {"role": "assistant", "content": "回答2"}
  ]
}
```

## CLI 对话

```bash
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
```

推理配置示例（`examples/inference/qwen3_lora_sft.yaml`）：

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
adapter_name_or_path: saves/qwen3-4b/lora/sft
template: qwen3_nothink
infer_backend: huggingface
```

## 内存管理

FastAPI 服务在 HuggingFace 后端下会启动 GPU 内存清理协程（`sweeper()`），每 300 秒调用 `torch_gc()` 释放未使用的 GPU 缓存。

## 关键文件

| 文件 | 说明 |
|------|------|
| `chat/chat_model.py` | ChatModel 门面类 |
| `chat/base_engine.py` | 引擎抽象接口 |
| `chat/hf_engine.py` | HuggingFace 推理 |
| `chat/vllm_engine.py` | vLLM 推理 |
| `chat/sglang_engine.py` | SGLang 推理 |
| `api/app.py` | FastAPI 应用创建与启动 |
| `api/chat.py` | 对话/打分请求处理 |
| `api/protocol.py` | OpenAI 兼容 Pydantic 模型 |
| `api/common.py` | 共享工具函数 |
