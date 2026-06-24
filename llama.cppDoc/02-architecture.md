# 02 - 整体架构

## 1. 分层架构

```
+----------------------------------------------------------+
|                    应用层 (tools/)                        |
|  llama-cli | llama-server | llama-quantize | mtmd | ...  |
+----------------------------------------------------------+
|                    公共层 (common/)                       |
|  chat模板 | 采样 | HF下载 | speculative | grammar | jinja |
+----------------------------------------------------------+
|                    核心层 (src/ + include/)               |
|  model | context | graph | kv-cache | sampler | vocab    |
+----------------------------------------------------------+
|                    模型层 (src/models/)                   |
|  llama | qwen3 | deepseek2 | mamba | rwkv7 | ... (134)   |
+----------------------------------------------------------+
|                    计算层 (ggml/)                         |
|  tensor ops | backend scheduler | quant kernels          |
+----------------------------------------------------------+
|                    后端层 (ggml/src/ggml-*)               |
|  cpu | cuda | metal | vulkan | sycl | hip | opencl | ... |
+----------------------------------------------------------+
|                    格式层 (gguf)                          |
|  GGUF 文件读写 | metadata | 权重存储                      |
+----------------------------------------------------------+
```

## 2. 推理数据流

```
用户输入 (文本/图像)
    |
    v
[Tokenization]  llama_tokenize()  ->  llama_vocab
    |
    v
[Batch 构建]    llama_batch       ->  llama_batch_allocr
    |
    v
[Memory 分配]   llama_memory_i    ->  llama_kv_cache / recurrent / hybrid
    |
    v
[Graph 构建]    llm_graph_context ->  ggml_cgraph (按架构)
    |
    v
[Graph 执行]    ggml_backend_sched ->  CPU/GPU kernels
    |
    v
[Logits 输出]   llama_get_logits()
    |
    v
[Sampling]      llama_sampler     ->  下一个 token
    |
    v
[Detokenization] llama_token_to_piece()
    |
    v
输出文本
```

## 3. 核心对象关系

```
llama_model                    # 模型权重 + 超参 (只读, 可共享)
    |
    +-- llama_vocab            # 分词器
    +-- llama_hparams          # 超参数
    +-- ggml tensors           # 权重张量
    |
    v
llama_context                  # 推理上下文 (每实例一个)
    |
    +-- llama_cparams          # 运行时参数 (ctx_size, n_batch, ...)
    +-- llama_memory_t         # KV cache / 状态记忆
    +-- ggml_backend_sched     # 多设备调度器
    +-- llama_sampler          # 采样器链
    |
    v
llama_batch                    # 输入批次
    |
    +-- tokens[]               # token ID 数组
    +-- pos[]                  # 位置数组
    +-- seq_id[]               # 序列 ID (多序列)
    +-- logits[]               # 是否需要 logits
```

## 4. 计算图系统

llama.cpp 使用 **静态计算图** 模式：

1. **构建阶段**: 根据模型架构和输入 shape 构建 `ggml_cgraph`
2. **复用阶段**: 相同 shape 的 decode 步可跳过重建 (`can_reuse()`)
3. **执行阶段**: `ggml_backend_sched` 将图分配到各 backend 执行

### 图类型 (`llm_graph_type`)

| 类型 | 用途 |
|------|------|
| `LLM_GRAPH_TYPE_DEFAULT` | 标准 decoder 前向 |
| `LLM_GRAPH_TYPE_ENCODER` | 编码器 (BERT/T5/多模态) |
| `LLM_GRAPH_TYPE_DECODER` | 解码器 (带 cross-attention) |
| `LLM_GRAPH_TYPE_DECODER_MTP` | 多 token 预测 (EAGLE3 等) |

### 图输入抽象

```
llm_graph_input_i (接口)
    |
    +-- llm_graph_input_embd      # token embeddings
    +-- llm_graph_input_pos       # 位置编码
    +-- llm_graph_input_attn      # attention mask
    +-- llm_graph_input_kv        # KV cache 指针
    +-- llm_graph_input_rs        # recurrent state (Mamba/RWKV)
    +-- llm_graph_input_cross     # cross-attention (多模态)
```

## 5. Memory 系统

不同模型架构使用不同的记忆机制：

| 类型 | 类 | 适用模型 |
|------|-----|----------|
| 标准 KV Cache | `llama_kv_cache` | Transformer (LLaMA, Qwen, ...) |
| ISWA KV Cache | `llama_kv_cache_iswa` | Sliding Window Attention |
| DSA KV Cache | `llama_kv_cache_dsa` | DeepSeek Sparse Attention |
| Recurrent | `llama_memory_recurrent` | Mamba, RWKV |
| Hybrid | `llama_memory_hybrid` | Jamba (Attn + SSM) |
| Hybrid ISWA | `llama_memory_hybrid_iswa` | Granite Hybrid |

所有类型实现统一接口 `llama_memory_i`：

```cpp
struct llama_memory_i {
    virtual llama_memory_context_ptr init_batch(...) = 0;
    virtual bool update(...) = 0;
    virtual void clear(...) = 0;
    virtual bool seq_rm(...) = 0;
    virtual void seq_cp(...) = 0;
    // ...
};
```

## 6. Backend 调度

```
ggml_backend_sched
    |
    +-- ggml_backend_cpu     # 默认 CPU 后端
    +-- ggml_backend_cuda    # GPU 层 offload
    +-- ggml_backend_metal   # Apple GPU
    +-- ...
    |
    v
Tensor Placement:
    - 权重按 n_gpu_layers 分配到 GPU
    - KV cache 可选 GPU offload
    - 中间激活值在对应 backend 计算
    - 跨 backend 自动插入 copy 节点
```

## 7. Server 架构

```
HTTP Request (OpenAI/Anthropic 格式)
    |
    v
server-http.cpp          # httplib 路由
    |
    v
server-queue.cpp         # 请求队列 + 连续批处理
    |
    v
server-context.cpp       # 推理上下文管理 (slots)
    |
    v
server-chat.cpp          # Chat 模板 + API 格式转换
    |
    v
libllama (decode + sample)
    |
    v
HTTP Response (SSE stream)
```

## 8. 模型加载流程

```
GGUF 文件
    |
    v
llama_model_loader       # 解析 GGUF header + metadata
    |
    v
llama_arch               # 识别架构 (llm_arch enum)
    |
    v
llama_model_xxx          # 架构特定: load_arch_hparams + load_arch_tensors
    |
    v
ggml_backend_buffer      # 权重分配到 CPU/GPU buffer
    |
    v
llama_model (ready)
```

## 9. CMake 构建依赖图

```
llama.cpp (root CMakeLists.txt)
    |
    +-- ggml/              -> libggml
    +-- src/               -> libllama (links ggml)
    +-- common/            -> libllama-common (links llama)
    +-- vendor/cpp-httplib
    +-- tools/
    |       +-- server/    -> llama-server (links common + mtmd)
    |       +-- cli/       -> llama-cli
    |       +-- ...
    +-- examples/          -> 各示例可执行文件
    +-- tests/             -> 单元测试
```
