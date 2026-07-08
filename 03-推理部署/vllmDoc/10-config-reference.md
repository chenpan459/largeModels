# 10 - 配置参考（VllmConfig）

统一配置：`vllm/config.py`（3900+ 行）

CLI 桥梁：`vllm/engine/arg_utils.py` → `EngineArgs` / `AsyncEngineArgs`

环境变量默认值：`vllm/envs.py`

## 配置创建流程

```
CLI args / LLM(...) kwargs
  → EngineArgs / AsyncEngineArgs
  → create_engine_config()
      1. create_model_config()
      2. _is_v1_supported_oracle() → 决定 VLLM_USE_V1
      3. _set_default_args_v1() 或 _set_default_args_v0()
      4. 组装 VllmConfig 各子 config
  → VllmConfig
  → VllmConfig.compute_hash()  # CUDA graph / compile 缓存键
```

配置优先级：**CLI > 环境变量 > 默认值**

## VllmConfig 聚合

```python
@dataclass
class VllmConfig:
    model_config: ModelConfig
    cache_config: CacheConfig
    parallel_config: ParallelConfig
    scheduler_config: SchedulerConfig
    device_config: DeviceConfig
    load_config: LoadConfig
    lora_config: Optional[LoRAConfig]
    speculative_config: Optional[SpeculativeConfig]
    quant_config: Optional[QuantizationConfig]
    compilation_config: CompilationConfig
    decoding_config: DecodingConfig
    observability_config: ObservabilityConfig
    kv_transfer_config: Optional[KVTransferConfig]   # V0 分离式 prefill
    prompt_adapter_config: Optional[PromptAdapterConfig]  # V0
    ...
```

`compute_hash()` 纳入：version、`VLLM_USE_V1`、model/cache/parallel/scheduler/device/load/quant/compilation 等——修改这些字段会使 CUDA graph 缓存失效。

## ModelConfig

| 字段 | 含义 |
|------|------|
| `model` | HF model id 或本地路径 |
| `tokenizer` | tokenizer 名（默认同 model） |
| `tokenizer_mode` | auto/slow/mistral |
| `dtype` | auto/float16/bfloat16/float32 |
| `max_model_len` | 最大序列长（可截断 hf_config） |
| `trust_remote_code` | 执行 HF 自定义 modeling 代码 |
| `runner_type` | `generate` / `pooling` |
| `hf_config` | Transformers PretrainedConfig |
| `is_multimodal` | 是否多模态 |
| `is_v1_compatible` | V1 兼容性标志 |

## CacheConfig

| 字段 | 默认 | 含义 |
|------|------|------|
| `block_size` | 16 | KV block token 数 |
| `gpu_memory_utilization` | 0.9 | KV 可用显存比例 |
| `enable_prefix_caching` | V1: True | 前缀 block 复用 |
| `prefix_caching_hash_algo` | builtin | hash 算法 |
| `swap_space` | 4 GiB | CPU swap（V0 为主） |
| `cache_dtype` | auto | KV dtype（含 fp8_e4m3 等） |

详见 [05-kv-cache-paged-attention.md](./05-kv-cache-paged-attention.md)。

## SchedulerConfig

| 字段 | 含义 |
|------|------|
| `max_num_batched_tokens` | 单 step 最大 token 总数 |
| `max_num_seqs` | 最大并发 sequence |
| `max_model_len` | 与 model 对齐 |
| `enable_chunked_prefill` | V1 强制 True |
| `long_prefill_token_threshold` | 长 prefill 单 step 上限（0=不限） |
| `scheduler_cls` | V1 默认 `vllm.v1.core.sched.scheduler.Scheduler` |
| `num_scheduler_steps` | V0 multi-step（V1 不支持） |
| `delay_factor` | V0 调度延迟（V1 不支持） |
| `policy` | V0 FCFS/priority（V1 不支持） |

### V1 硬件默认（`_set_default_args_v1`）

| GPU | max_num_batched_tokens (serve) | max_num_seqs |
|-----|-------------------------------|--------------|
| H100/H200 | 8192 | 1024 |
| 其他 | 2048 | 256 |

LLM class API 的 batched_tokens 上限更高（16384 on H100）。

## ParallelConfig

| 字段 | 含义 |
|------|------|
| `tensor_parallel_size` | TP |
| `pipeline_parallel_size` | PP |
| `data_parallel_size` | DP |
| `distributed_executor_backend` | `mp` / `ray` / `uni` |
| `world_size` | TP × PP × DP |

## LoadConfig

| 字段 | 含义 |
|------|------|
| `load_format` | auto/safetensors/pt/gguf/tensorizer/... |
| `download_dir` | HF 缓存目录 |
| `ignore_patterns` | 跳过的 weight 文件 pattern |

## LoRAConfig

| 字段 | 含义 |
|------|------|
| `max_loras` | 最大同时加载 adapter 数 |
| `max_lora_rank` | 最大 rank |
| `lora_dtype` | adapter 权重 dtype |
| `max_cpu_loras` | CPU 缓存 adapter 数 |

## SpeculativeConfig

| 字段 | 含义 |
|------|------|
| `method` | ngram / eagle / medusa / ... |
| `model` | draft 模型路径 |
| `num_speculative_tokens` | 每 step draft 数 |
| `draft_tensor_parallel_size` | draft 模型 TP |

V1 oracle 仅支持 **ngram**、**eagle**。

## QuantizationConfig

由 `--quantization awq/gptq/fp8/...` 解析：

- 指向 `model_executor/layers/quantization/` 中具体 `QuantizationConfig` 子类
- 层间应用不同 kernel（Marlin、Triton、Cutlass）

详见 [15-quantization-catalog.md](./15-quantization-catalog.md)。

## CompilationConfig

| 字段 | 含义 |
|------|------|
| `level` | CompilationLevel 0-3 |
| `use_cudagraph` | 是否 capture CUDA graph |
| `cudagraph_capture_sizes` | capture batch size 列表 |
| `use_inductor` | torch inductor backend |
| `custom_ops` | 允许 custom op 列表 |
| `splitting_ops` | 子图拆分边界 |

### CompilationLevel

| Level | 名称 | 行为 |
|-------|------|------|
| 0 | `NO_COMPILATION` | Eager（`--enforce-eager`） |
| 1 | `DYNAMO_AS_IS` | torch.compile 整图 |
| 2 | `DYNAMO_ONCE` | compile 一次 |
| 3 | `PIECEWISE` | 子图 + CUDA graph（V1 decode 默认路径） |

## DecodingConfig

| 字段 | 含义 |
|------|------|
| `guided_decoding_backend` | xgrammar / guidance / auto |
| `disable_fallback` | 禁用 fallback |

V1 structured output 用 grammar bitmask，不用 V0 logits processor。

## ObservabilityConfig

| 字段 | 含义 |
|------|------|
| `show_hidden_metrics` | 隐藏指标 |
| `otlp_traces_endpoint` | OTLP tracing（V1 不支持） |

## KVTransferConfig（V0）

分离式 prefill / disaggregated serving：

- KV 跨节点传输
- V1 oracle 当前拒绝

## 环境变量（完整重要项）

| 变量 | 默认 | 含义 |
|------|------|------|
| `VLLM_USE_V1` | **True** | V0/V1 选择 |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | True | 同步 LLM 子进程 |
| `VLLM_ATTENTION_BACKEND` | None | 强制 attention backend |
| `VLLM_USE_FLASHINFER_SAMPLER` | None | FlashInfer top-k/top-p |
| `VLLM_LOGGING_LEVEL` | INFO | 日志级别 |
| `VLLM_WORKER_MULTIPROC_METHOD` | spawn | Worker 进程启动方式 |
| `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` | 128 | 输出处理 chunk |
| `VLLM_MLA_DISABLE` | False | 禁用 MLA |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | False | 热更新 LoRA |
| `VLLM_DISABLE_COMPILE_CACHE` | False | 禁用 compile 缓存 |

完整列表见 `envs.py`。

## EngineArgs 用法

```python
from vllm.engine.arg_utils import AsyncEngineArgs

args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=2,
    max_num_seqs=256,
    gpu_memory_utilization=0.95,
    enable_prefix_caching=True,
)
vllm_config = args.create_engine_config()
```

同步 API 用 `EngineArgs`；OpenAI server 用 `AsyncEngineArgs`。

## 配置示例

### 高并发服务

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --tensor-parallel-size 1
```

### 长上下文

```bash
vllm serve model \
  --max-model-len 131072 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.9
```

（需模型与 GPU 显存支持。）

### AWQ 量化

```bash
vllm serve model --quantization awq
```

### 强制 V0

```bash
VLLM_USE_V1=0 vllm serve model
```

### 调试 Eager

```bash
vllm serve model --enforce-eager --max-num-seqs 4
```

## Platform 侧配置修改

`platforms/cuda.py` → `check_and_update_config()`：

- 自动设置 `worker_cls`
- MLA 时 `block_size=64`
- 不支持特性时 warning/error

## 阅读源码

```bash
# 列出所有 config dataclass
grep -n "^class.*Config" vllm/config.py
grep -n "^@dataclass" vllm/config.py
```

改配置时注意 `compute_hash()` 注释——影响 graph 的字段需纳入 hash。

## 相关文档

- V0/V1 选择：[14-v0-v1-migration.md](./14-v0-v1-migration.md)
- KV cache：[05-kv-cache-paged-attention.md](./05-kv-cache-paged-attention.md)
- 量化：[15-quantization-catalog.md](./15-quantization-catalog.md)
