# 12 - llama-server HTTP 服务

## 1. 模块概述

**llama-server** 是 llama.cpp 的 HTTP 推理服务，提供 OpenAI/Anthropic 兼容的 REST API 和内置 Web UI。支持多用户并行、连续批处理、多模态、投机解码等企业级特性。

- **路径**: `tools/server/`
- **HTTP 库**: cpp-httplib
- **JSON 库**: nlohmann/json
- **Web UI**: `tools/ui/` (可选嵌入)

## 2. 源文件结构

```
tools/server/
├── main.cpp              # 入口
├── server.cpp            # 服务主逻辑
├── server-http.cpp/h     # HTTP 路由与请求处理
├── server-context.cpp/h  # 推理上下文 (slot) 管理
├── server-queue.cpp/h    # 请求队列 + 连续批处理
├── server-chat.cpp/h     # Chat API 格式转换
├── server-task.cpp/h     # 异步任务管理
├── server-models.cpp/h   # 多模型路由
├── server-tools.cpp/h    # Function calling
├── server-schema.cpp/h   # JSON Schema 验证
├── server-common.cpp/h   # 共享工具
├── server-cors-proxy.h   # CORS 代理
├── tests/                # Python 集成测试
└── bench/                # 性能基准
```

### 构建产物

| Target | 类型 | 说明 |
|--------|------|------|
| `server-context` | 静态库 | 核心服务逻辑 |
| `llama-server-impl` | 库 | HTTP + 模型管理 |
| `llama-server` | 可执行 | HTTP 服务入口 |

## 3. 架构

```
                    HTTP Clients
                         |
                         v
              +---------------------+
              |  server-http.cpp    |
              |  (httplib routes)   |
              +---------------------+
                         |
              +---------------------+
              |  server-queue.cpp   |
              |  (请求队列/调度)     |
              +---------------------+
                         |
              +---------------------+
              | server-context.cpp  |
              | (slot 管理)          |
              |  Slot 0 | Slot 1 |..|
              +---------------------+
                         |
              +---------------------+
              |  server-chat.cpp    |
              |  (格式转换)          |
              +---------------------+
                         |
              +---------------------+
              |  libllama           |
              |  decode + sample    |
              +---------------------+
```

## 4. API 端点

### 4.1 OpenAI 兼容

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | Chat 补全 (流式/非流式) |
| `/v1/completions` | POST | 文本补全 |
| `/v1/embeddings` | POST | 文本嵌入 |
| `/v1/models` | GET | 模型列表 |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/rerank` | POST | 重排序 |

### 4.2 Anthropic 兼容

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/messages` | POST | Anthropic Messages API |

### 4.3 扩展端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/tokenize` | POST | 分词 |
| `/detokenize` | POST | 反分词 |
| `/apply-template` | POST | 应用 Chat 模板 |
| `/infill` | POST | 代码填充 (FIM) |
| `/slots` | GET/POST | Slot 管理 |
| `/props` | GET | 服务属性 |
| `/health` | GET | 健康检查 |
| `/metrics` | GET | Prometheus 指标 |

### 4.4 多模态

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 支持 image_url content |

## 5. 核心特性

### 5.1 连续批处理 (Continuous Batching)

```
Time ->
Slot 0: [====prompt====][=gen=][=gen=][=gen=]
Slot 1:      [==prompt==][=gen=][=gen=]
Slot 2:           [===prompt===][=gen=]
                    ^ 同一 decode 步合并处理
```

- 多个用户请求合并为一个 micro-batch
- 新请求可在任意时刻加入
- 完成的序列立即释放 slot

### 5.2 Slot 系统

每个 slot 是独立的推理上下文：

```json
{
  "id": 0,
  "state": "processing",
  "n_ctx": 4096,
  "n_past": 128,
  "params": { "temperature": 0.8 }
}
```

- 支持 slot 保存/恢复 (KV cache 持久化)
- 支持 slot fork (复制对话)
- `--parallel N` 控制 slot 数量

### 5.3 流式输出 (SSE)

```
POST /v1/chat/completions
Content-Type: application/json
{"stream": true, "messages": [...]}

Response:
data: {"choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"content":" world"}}]}
data: [DONE]
```

### 5.4 Function Calling

```json
{
  "messages": [{"role": "user", "content": "What's the weather?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "parameters": {"type": "object", "properties": {...}}
    }
  }]
}
```

实现位于 `server-tools.cpp`，支持多种 tool call 格式。

### 5.5 投机解码

```bash
llama-server -m large-model.gguf -md draft-model.gguf --draft-max 16
```

- `-md`: draft 模型路径
- `--draft-max`: 最大 draft token 数
- `--draft-min`: 最小 draft token 数

### 5.6 多模型路由

```bash
llama-server --models-dir /path/to/models/
```

`server-models.cpp` 管理多模型加载和路由。

## 6. 启动与配置

### 6.1 基本启动

```bash
# 本地模型
llama-server -m model.gguf --host 0.0.0.0 --port 8080

# HuggingFace
llama-server -hf ggml-org/gemma-3-1b-it-GGUF

# 多模态
llama-server -m model.gguf --mmproj mmproj.gguf

# GPU
llama-server -m model.gguf -ngl 99 -c 8192 --parallel 4
```

### 6.2 关键参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--host` | 监听地址 | 127.0.0.1 |
| `--port` | 监听端口 | 8080 |
| `-c, --ctx-size` | 上下文长度 | 模型默认 |
| `--parallel` | 并行 slot 数 | 1 |
| `-b, --batch-size` | 最大 batch | 2048 |
| `-ub, --ubatch-size` | micro-batch | 512 |
| `-ngl` | GPU 层数 | 0 |
| `-fa, --flash-attn` | Flash Attention | auto |
| `--cont-batching` | 连续批处理 | true |
| `--metrics` | Prometheus 指标 | false |
| `--slots` | 启用 slot API | false |
| `--props` | 启用 props 端点 | false |

### 6.3 环境变量

| 变量 | 说明 |
|------|------|
| `LLAMA_ARG_THREADS` | CPU 线程数 |
| `LLAMA_ARG_CTX_SIZE` | 上下文长度 |
| `LLAMA_ARG_N_GPU_LAYERS` | GPU 层数 |
| `LLAMA_ARG_HOST` | 监听地址 |
| `LLAMA_ARG_PORT` | 监听端口 |

## 7. Web UI

Server 内置 Web UI (位于 `tools/ui/`)：

- 构建时嵌入 (`LLAMA_BUILD_UI=ON`)
- 或使用预构建版本 (`LLAMA_USE_PREBUILT_UI=ON`)
- 访问: `http://localhost:8080`

## 8. 测试

```bash
# 启动 server
llama-server -m model.gguf --port 8080 &

# 运行测试
cd tools/server/tests
pip install -r requirements.txt
pytest unit/test_chat_completion.py
```

测试覆盖：
- Chat completion (流式/非流式)
- Embeddings
- Tokenize/Detokenize
- Function calling
- Reranking
- 多模态 (vision)
- Speculative decoding
- Slot 管理
- 安全性

## 9. 性能基准

```bash
cd tools/server/bench
python bench.py --host localhost --port 8080
```

## 10. Docker 部署

```bash
docker run -p 8080:8080 -v /models:/models \
    ghcr.io/ggml-org/llama.cpp:server \
    -m /models/model.gguf --host 0.0.0.0 --port 8080
```

## 11. 扩展开发

| 需求 | 修改位置 |
|------|----------|
| 新 API 端点 | `server-http.cpp` 添加路由 |
| 新 API 格式 | `server-chat.cpp` |
| 请求调度逻辑 | `server-queue.cpp` |
| Slot 管理 | `server-context.cpp` |
| Tool calling | `server-tools.cpp` |
| 请求验证 | `server-schema.cpp` |

开发文档: `tools/server/README-dev.md`
