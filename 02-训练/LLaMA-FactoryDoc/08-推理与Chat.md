# 08 — 推理与 Chat 源码分析

> 对应源码版本：LLaMA Factory `0.9.6.dev0`。核心位于 `src/llamafactory/chat/` 与 `src/llamafactory/api/`。

## 1. 总体结构

```text
CLI chat / WebUI / FastAPI / Python 调用
  → ChatModel                         同步/异步统一门面
    → BaseEngine 异步接口
      → HuggingfaceEngine             本进程 Transformers 模型
      → VllmEngine                    本进程 AsyncLLMEngine
      → SGLangEngine                  子进程 HTTP server
```

`BaseEngine` 要求实现三个异步方法：

```text
chat(...)        → list[Response]
stream_chat(...) → AsyncGenerator[str]
get_scores(...)  → list[float]
```

`Response` 精确字段为 `response_text`、`response_length`、`prompt_length`、`finish_reason("stop"|"length")`。

## 2. `ChatModel` 的同步/异步桥

构造函数先调用 `get_infer_args()`，再按 `model_args.infer_backend` 选择：

- `"huggingface"` → `HuggingfaceEngine`；
- `"vllm"` → `VllmEngine`；
- `"sglang"` → `SGLangEngine`。

之后无条件创建一个新的 asyncio event loop，并用 daemon `Thread` 执行 `loop.run_forever()`。这个后台 loop 专门为同步 API 服务。

### 非流式桥接

```text
chat()
  → run_coroutine_threadsafe(self.achat(...), self._loop)
  → Future.result() 阻塞调用线程
  → achat()
  → await engine.chat(...)
```

`get_scores()/aget_scores()` 同理。

### 流式桥接

```text
stream_chat()
  → 创建 self.astream_chat(...) 异步生成器
  → 每次把 generator.__anext__() 提交给后台 loop
  → task.result()
  → 捕获 StopAsyncIteration
```

原生异步调用 `achat/astream_chat/aget_scores` 不经过后台 loop，直接 await engine。FastAPI 使用的正是这组异步方法。

注意：对象没有显式 `close()`，后台 loop 线程随进程退出；大量反复创建 `ChatModel` 会累计 daemon 线程。应用通常应复用单例。

## 3. Hugging Face 后端

### 初始化

`HuggingfaceEngine`：

1. `can_generate = (stage == "sft")`；
2. 加载 tokenizer/processor；
3. 生成模型使用左 padding，奖励模型使用右 padding；
4. 初始化 template；
5. `load_model(..., is_trainable=False, add_valuehead=not can_generate)`；
6. 保存 generation args；
7. 创建 `asyncio.Semaphore(int(MAX_CONCURRENT or 1))`。

只有 HF 后端支持 `get_scores()`；stage 非 SFT 时加载 value head，chat/stream_chat 会拒绝调用。

### 输入处理与生成

`_process_args()` 会原地补齐缺失的媒体占位符到第一条消息，然后：

```text
template.mm_plugin.process_messages()
→ messages + 空 assistant
→ template.encode_oneturn()
→ mm_plugin.process_token_ids()
→ input_ids + attention_mask
→ GenerationConfig
→ mm_plugin.get_mm_inputs()
```

停止 token 来自 `template.get_stop_token_ids()`。请求参数覆盖 YAML generation args；`max_length` 与 `max_new_tokens` 互斥。`num_return_sequences > 1` 会强制采样。HF 后端当前明确忽略 `stop` 字符串并记录 warning。

非流式 `_chat()` 直接 `model.generate()`，切掉 prompt 后 batch decode，并根据 EOS 是否出现设置 finish reason。异步层用 `asyncio.to_thread()` 避免阻塞 event loop。

流式 `_stream_chat()` 使用 `TextIteratorStreamer`，另起 daemon 线程执行 `model.generate()`；异步方法再反复 `asyncio.to_thread(streamer.__next__)`。所以 HF 流式涉及两层线程：模型生成线程与异步适配线程。

### 奖励打分

`_get_scores()` 对字符串 batch 做右 padding/truncation，调用 value-head model，取每条 attention mask 最后一个有效 token 的 value。`max_length` 默认取 config 的 `max_position_embeddings`，没有则 1024。

`MAX_CONCURRENT` 只是并发信号量，不会自动做 continuous batching；提高它可能增加显存峰值，且同一模型的多线程 generate 是否合适取决于模型实现。

## 4. vLLM 后端

初始化只加载 config、tokenizer、processor、template，不调用 LLaMA Factory 的 `load_model()`。模板多模态 plugin 的 `expand_mm_tokens=False`，token 扩展交给 vLLM。

核心 engine 参数：

```text
model, trust_remote_code, download_dir, dtype,
max_model_len=vllm_maxlen,
tensor_parallel_size=get_device_count() or 1,
gpu_memory_utilization,
enforce_eager,
enable_lora,
max_lora_rank
```

用户字典 `vllm_config` 最后更新参数，可覆盖上述值。创建方式是：

```text
AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
```

存在 adapter 时只使用 `adapter_name_or_path[0]` 创建一个 `LoRARequest`；这不是模型模块的多 adapter 合并路径。

`_generate()` 使用 template 编码 prompt，构造 vLLM `SamplingParams`，支持 request-level `stop` 和 template stop token IDs。图片/视频/音频经 plugin 的 regularize 方法写入 `multi_modal_data`，然后调用异步 engine `generate()`。

`chat()` 消费完整 async iterator 并读取最后输出；`stream_chat()` 比较累计文本，yield 增量。vLLM 不支持 `get_scores()`。`length_penalty` 当前只 warning，不生效。

模型 config 为 GPTQ 且 `infer_dtype=auto` 时强制 fp16。多模态默认限制每个 prompt 最多 image 4、video 2、audio 2，可通过 `vllm_config` 调整。

## 5. SGLang 后端

SGLang 也只由 LLaMA Factory 编码 prompt，但执行模型位于外部子进程：

1. 拼出 `python3 -m sglang.launch_server` 命令；
2. 设置 model path、dtype、context length、静态显存比例、TP size、download dir；
3. `launch_server_cmd()` 启动；
4. `wait_for_server(..., timeout=300)`；
5. 记录 `base_url`，通过 `atexit` 和 `__del__` 清理。

LoRA 只加载第一个 adapter，服务参数包含 `--max-loras-per-batch 1`、`lora0=<path>`，并禁用 radix cache。

每次请求向子服务 `/generate` 发送：

```text
input_ids
sampling_params
stream=True
lora_request=["lora0"]（可选）
```

底层使用同步 `requests.post(..., stream=True)`，但 `_generate()` 通过 `asyncio.to_thread()` 创建迭代器。返回的是普通 Python generator，因此 `chat/stream_chat` 使用普通 `for` 消费，不是 `async for`。这意味着响应流迭代本身仍可能在调用 event loop 的线程中阻塞；高并发时要实测延迟。

SGLang 只支持 `num_return_sequences=1`，不支持 score。GPTQ auto dtype 同样强制 fp16。媒体先经过 template plugin 修改消息，但该 backend 请求只发送 `input_ids`，源码没有像 vLLM 一样传 `multi_modal_data`，因此不要仅凭接口参数判断多模态可用性。

## 6. 三后端能力差异

| 能力 | Hugging Face | vLLM | SGLang |
|---|---|---|---|
| SFT chat | 支持 | 支持 | 支持 |
| RM score | 支持 | 不支持 | 不支持 |
| 流式 | TextIteratorStreamer | AsyncLLMEngine | 子服务 SSE |
| request `stop` | 不支持，仅 warning | 支持 | 支持 |
| `n > 1` | 支持并强制采样 | 支持 | 不支持 |
| LoRA | 走模型模块加载/合并 | 仅首个 LoRARequest | 仅首个 lora0 |
| 并发控制 | `MAX_CONCURRENT` semaphore | vLLM scheduler | SGLang server |
| 多模态载荷 | plugin → model kwargs | `multi_modal_data` | 当前仅 prompt IDs |
| tensor parallel | 不由 ChatModel 自动开启 | 默认全部可见设备 | `sglang_tp_size` 或全部设备 |

## 7. FastAPI 调用链

`run_api()`：

```text
ChatModel()
→ create_app(chat_model)
→ uvicorn.run(host=API_HOST or 0.0.0.0,
              port=API_PORT or 8000)
```

`create_app()` 读取 `FASTAPI_ROOT_PATH`，安装允许任意 origin/method/header 的 CORS，并用可选 `API_KEY` 做 Bearer 校验。模型名只用于协议响应，来自 `API_MODEL_NAME`，默认 `"gpt-3.5-turbo"`。

端点：

- `GET /v1/models`；
- `POST /v1/chat/completions`；
- `POST /v1/score/evaluation`。

生成模型访问 score 返回 405；奖励模型访问 chat 返回 405。HF 后端 lifespan 每 300 秒执行 `torch_gc()`；其他后端不启用 sweeper，应用关闭时仍统一清理一次。

## 8. API 请求标准化

`_process_request()` 会修改 `request.messages`：若首条是 system，就 `pop(0)`。之后要求剩余消息数为奇数，严格支持：

```text
user/tool → assistant/function → user/tool → ...
```

assistant 的 `tool_calls` 转为内部 function JSON；OpenAI 多模态 content item 转为文本占位符和媒体对象。支持：

- image：base64、受检查的本地路径、经过 SSRF 检查的 URL；
- video：base64、本地路径、URL；
- audio：base64、本地路径、URL。

远程媒体获取使用同步 `requests.get()`，且发生在 FastAPI async handler 内，可能阻塞 event loop。工具定义序列化为 template 接收的 JSON 字符串。

`ChatCompletionRequest` 的实际字段为：

```text
model, messages, tools,
do_sample, temperature, top_p, n,
presence_penalty, max_tokens, stop, stream
```

其中 `presence_penalty` 被映射为 engine 的 `repetition_penalty`，语义并非 OpenAI presence penalty 的精确实现。

Score 请求不是对话消息对象，协议是：

```json
{"model": "reward-model", "messages": ["完整文本1", "完整文本2"], "max_length": 2048}
```

## 9. FastAPI 流式约束

`stream=true` 返回 `EventSourceResponse`。生成器先发空 assistant role chunk，再逐增量文本发送 chunk，最后发送 `finish_reason="stop"` 和 `[DONE]`。

源码在进入 engine 前强制：

- 有 `tools` 时返回 400：`Cannot stream function calls.`；
- `n > 1` 时返回 400：`Cannot stream multiple responses.`。

因此非流式才能返回结构化 tool calls 或多个 choice。流式终止原因固定写 STOP，没有传播 backend 的 length finish reason，也不返回 usage。

非流式完成后，如果请求带 tools，会调用 `template.extract_tool()`；解析为函数列表时生成 OpenAI tool_calls 并设 `finish_reason=tool_calls`，否则返回普通 content。usage 取 engine 返回的 token 数。

## 10. CLI Chat

`run_chat()` 创建一个 `ChatModel` 和内存 messages 列表。每轮：

1. 添加 user；
2. 调用同步 `stream_chat()`；
3. 累积完整 assistant 文本；
4. 写回 history。

`clear` 清空历史并 `torch_gc()`，`exit` 退出。它不处理 tools 和媒体，仅是同步桥最直接的调用示例。

## 11. 扩展点与陷阱

- 新 backend：继承 `BaseEngine`，实现三个异步方法，在 `ChatModel.__init__` 注册，并同步更新参数校验。
- 新协议字段：从 `api/protocol.py` 到 `api/chat.py` 再到各 engine 参数映射需要端到端修改。
- 新工具格式：由 Template 的 `format_tools/extract_tool` 扩展，不应硬编码在 engine。
- 多模态：必须同时实现 API 解码、template plugin 与 backend 载荷传递。

常见陷阱：

1. 后端值是 `huggingface/vllm/sglang`，不是 `hf`。
2. 只有 HF 支持 RM score；参数解析会明确禁止 vLLM 的非 SFT stage，而 SGLang 即便以非 SFT stage 构造也既不能 chat、也没有 score 实现。
3. HF 的 `stop` 当前不生效；API 接受该字段不等于所有 backend 支持。
4. 流式 tools 和流式多 choice 被 FastAPI 明确拒绝。
5. API 要求严格交替，不能发送连续 user 或偶数条非 system 消息。
6. API 的远程媒体下载和 SGLang response iteration 含同步 I/O，高并发部署需关注 event-loop 阻塞。
7. vLLM/SGLang 只取第一个 adapter；与 HF 模型模块的 adapter 合并规则不同。
8. `ChatModel` 同步方法不能被误认为原生 async；它通过后台线程 loop 阻塞等待结果。
9. CORS 默认完全开放；启用公网服务时应在网关或源码中收紧，并设置 `API_KEY`。
