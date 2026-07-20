# 06 - Model Executor 与 GPUModelRunner

## model_executor 概览

路径：`vllm/model_executor/`（~598 文件）

```
model_executor/
├── model_loader/
│   ├── __init__.py          # get_model()
│   ├── loader.py            # DefaultModelLoader、权重迭代
│   └── weight_utils.py      # HF safetensors、S3、shard
├── models/
│   ├── registry.py          # 架构名 → 类映射（核心）
│   ├── llama.py, qwen2.py, deepseek_v2.py
│   ├── mixtral.py, qwen3_moe.py, mllama.py
│   └── ...（100+ 架构）
├── layers/
│   ├── linear.py            # ColumnParallelLinear、RowParallelLinear
│   ├── rotary_embedding.py  # RoPE、M-RoPE
│   ├── fused_moe.py         # MoE grouped GEMM
│   ├── logits_processor.py  # TP vocab gather
│   ├── sampler.py           # get_sampler() 分发 V0/V1
│   ├── vocab_parallel_embedding.py
│   └── quantization/        # 72 文件，见 15-量化方案目录.md
└── parameter.py             # ModelWeightParameter、Packed参数
```

## 模型注册（Registry）

`models/registry.py` — 多字典合并为 `_VLLM_MODELS`：

| 字典 | 用途 |
|------|------|
| `_TEXT_GENERATION_MODELS` | 标准 causal LM |
| `_EMBEDDING_MODELS` | embedding 模型 |
| `_MULTIMODAL_MODELS` | VLM（LLaVA、Qwen-VL 等） |
| `_CROSS_ENCODER_MODELS` | rerank/score |
| `_SPECULATIVE_DECODING_MODELS` | draft 模型 |
| `_TRANSFORMERS_MODELS` | Transformers backend 通用 |

```python
# 加载时
architecture = get_model_architecture(hf_config)
model_cls = ModelRegistry.resolve_model_cls(architecture)
```

扩展新模型：

1. 实现 `nn.Module` forward（含 `Attention` 层）
2. 在 registry 注册 `architectures` 字符串
3. 可选：自定义 `transformers_utils/configs/` 解析 HF config
4. 设置 `is_v1_compatible` 标志（Mamba、enc-dec 等可能为 False）

## 权重加载流程

```python
from vllm.model_executor.model_loader import get_model

loader = get_model_loader(vllm_config.load_config)
model = loader.load_model(vllm_config=vllm_config)
```

`model_loader/loader.py` 流程：

```
1. 解析 model path（HF Hub / 本地 / S3）
2. get_model_architecture() → registry lookup
3. 初始化空模型（meta device 或 CPU）
4. configure_quant_config() → 各层 QuantizeMethod
5. 按 TP rank 迭代 shard 加载权重
6. process_weights_after_loading() → Marlin repack 等
7. model.eval() + 移到 target device
```

### LoadFormat

| Format | 说明 |
|--------|------|
| `auto` | 自动检测 |
| `safetensors` / `pt` | 标准 HF |
| `gguf` | GGUF（**V1 不支持**） |
| `tensorizer` | CoreWeave 快速加载（**V1 不支持**） |
| `runai_streamer` | Run:ai 流式加载 |
| `dummy` | 随机权重（测试） |

TP 分片：每 rank 只加载 `tensor_model_parallel_rank` 对应的 column/row shard。

## GPUModelRunner

文件：`vllm/v1/worker/gpu_model_runner.py`（~1700 行）

**单 GPU 上的执行单元**，由 `GPUWorker`（`gpu_worker.py`）持有。

### 初始化

```
1. get_model(vllm_config)           # 加载 transformer
2. get_attn_backend()               # 选择 attention backend
3. 分配 KV cache tensors
4. bind_kv_cache()
5. Sampler / RejectionSampler
6. Spec decode proposer（若启用）
7. CUDA graph capture（若 compilation_config 允许）
8. InputBatch 预分配
```

### execute_model() 详细流程

```python
def execute_model(self, scheduler_output) -> ModelRunnerOutput:
    self._update_states(scheduler_output)
    if self.is_multimodal:
        self._execute_mm_encoder(scheduler_output)
    attn_metadata, logits_indices, spec_decode_metadata = \
        self._prepare_inputs(scheduler_output)
    with set_forward_context(attn_metadata, ...):
        hidden_states = self.model(input_ids, positions, ...)
    if not get_pp_group().is_last_rank:
        return IntermediateTensors(...)  # PP 中间 stage
    logits = self.model.compute_logits(sample_hidden_states)
    if grammar_bitmask is not None:
        logits = apply_grammar_bitmask(logits, grammar_bitmask)
    if spec_decode_metadata:
        output = self.rejection_sampler(...)
    else:
        output = self.model.sample(logits, sampling_metadata)
    return ModelRunnerOutput(...)
```

### _update_states() — Persistent Batch

```
1. 移除 finished_req_ids 对应 request
2. 移除本 step 未调度的 running request（保留在 self.requests 缓存）
3. 处理 preempted：替换 block table row
4. 添加 scheduled_new_reqs → CachedRequestState
5. 更新 scheduled_cached_reqs 的 token ids、block ids
6. 同步 InputBatch CPU/GPU buffer
```

**性能提示**（源码注释）：若连续 step 间 batch 重叠度低，persistent batch 优势减弱，可能出现 GPU 空转。

### _prepare_inputs()

构建：

- `FlashAttentionMetadata`（或 Triton/MLA 对应 metadata）
- `query_start_loc`、`seq_lens`、block table GPU tensor
- `SamplingMetadata`（temperature、top_p、top_k、penalties）
- `SpecDecodeMetadata`（draft token 布局）
- `logits_indices`：需要从 hidden states 取 logits 的 token 位置

### CUDA Graph

条件：

```python
compilation_config.level == CompilationLevel.PIECEWISE
and not enforce_eager
```

- `pad_for_cudagraph()` 将 token 数 pad 到 capture size
- `cudagraph_capture_sizes` 来自 config（如 [1,2,4,8,...,256]）
- decode 小 batch 受益最大

## ForwardContext

`forward_context.py` — 线程局部上下文：

```python
@contextmanager
def set_forward_context(attn_metadata, vllm_config, ...):
    _forward_context.attn_metadata = attn_metadata
    _forward_context.cudagraph_runtime_mode = ...
    yield
```

`Attention` 层在 forward 中读取 `_forward_context` 获取 metadata 和 graph 模式。

Pipeline Parallel：`IntermediateTensors` 在 stage 间传递 hidden states。

## InputBatch

`v1/worker/gpu_input_batch.py`：

| Buffer | 用途 |
|--------|------|
| `token_ids_cpu` / GPU | 本 step 输入 token |
| `num_computed_tokens_cpu` | 每 request 已算 token 数 |
| `block_table` | BlockTable 实例 |
| `temperature`, `top_p`, `top_k` | 采样参数 |
| `greedy_reqs`, `random_reqs` | 采样路径分流 |

构建 `SamplingMetadata` 供 `Sampler` 使用。

## 典型模型 forward（Llama）

`models/llama.py` 结构：

```python
class LlamaModel(nn.Module):
    embed_tokens → layers[N] × LlamaDecoderLayer → norm

class LlamaDecoderLayer:
    input_layernorm → Attention → post_attention_layernorm → MLP

class LlamaForCausalLM:
    model → compute_logits(hidden_states[special_indices])
         → sample(logits, sampling_metadata)  # 末 rank
```

`Attention` 层（`attention/layer.py`）统一调用 backend impl。

## MoE 执行

`layers/fused_moe.py` → `FusedMoE`：

- TopK router → token dispatch
- Grouped GEMM（Cutlass / Triton / DeepGemm）
- Expert Parallel：`distributed/device_communicators/` all-to-all

相关模型：`mixtral.py`、`qwen3_moe.py`、`deepseek_v2.py`、`dbrx.py`。

## LoRA

`v1/worker/lora_model_runner_mixin.py` → `LoRAModelRunnerMixin`：

```python
LRUCacheWorkerLoRAManager   # LRU 缓存 adapter 权重
LoRAMapping                  # request → adapter id
set_active_loras(input_batch)  # 每 step 激活
```

- Warmup rank：`LORA_WARMUP_RANK = 8`
- 运行时 load/unload：`EngineCore.add_lora()` → executor → workers
- API：`POST /v1/load_lora_adapter`、`/v1/unload_lora_adapter`
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING` 控制热更新

## Pooling / Embedding 模式

V0 路径（V1 不支持 generate 以外 runner_type）：

| Runner | 路径 |
|--------|------|
| `pooling_model_runner.py` | embedding/rerank/classify |
| `cpu_pooling_model_runner.py` | CPU |

API：`POST /v1/embeddings`、`/v1/rerank`、`/pooling`

## 其他硬件 Runner

| Runner | 路径 | 说明 |
|--------|------|------|
| `cpu_model_runner.py` | V0 CPU | |
| `tpu_model_runner.py` | TPU | Pallas backend |
| `neuron_model_runner.py` | AWS Neuron | |
| `xpu_model_runner.py` | Intel XPU | |

V1 主要维护 `gpu_model_runner.py`；其他平台跟进中。

## torch.compile 与 CompilationConfig

| Level | 名称 | 行为 |
|-------|------|------|
| 0 | `NO_COMPILATION` | Eager |
| 1 | `DYNAMO_AS_IS` | torch.compile 整图 |
| 2 | `DYNAMO_ONCE` | compile 一次 |
| 3 | `PIECEWISE` | 子图拆分 + CUDA graph |

字段：`use_cudagraph`、`cudagraph_capture_sizes`、`use_inductor`、`custom_ops`、`splitting_ops`。

`--enforce-eager` 强制 Level 0。

## Sampler 分发

`model_executor/layers/sampler.py`:

```python
def get_sampler(vllm_config):
    if envs.VLLM_USE_V1:
        return v1.sample.sampler.Sampler()
    return Sampler()  # V0
```

详见 [13-采样与结构化输出.md](./13-采样与结构化输出.md)。

## 阅读顺序

1. `models/llama.py` — 典型 decoder 结构
2. `attention/layer.py` — KV write + backend 调用
3. `gpu_model_runner.py` — `execute_model`、`_update_states`、`_prepare_inputs`
4. `gpu_input_batch.py` — batch 布局
5. 目标架构（如 `deepseek_v2.py`）— MLA + MoE
6. [15-量化方案目录.md](./15-量化方案目录.md) — 量化

## 与训练框架

| 训练 | 推理（vLLM） |
|------|--------------|
| Megatron TP/PP checkpoint | Bridge → HF safetensors → vLLM loader |
| LLaMA-Factory LoRA adapter | LoRA mixin + load_lora API |
| 权重格式 | HuggingFace（非 GGUF；GGUF 用 llama.cpp） |

## 关键源码行号

| 主题 | 位置 |
|------|------|
| execute_model | `gpu_model_runner.py:990-1109` |
| _update_states | `gpu_model_runner.py:317+` |
| get_model | `model_loader/__init__.py` |
| Registry | `models/registry.py` |
