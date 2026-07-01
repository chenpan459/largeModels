# 10 - 配置参考（VllmConfig）

统一配置：`vllm/config.py`（3900+ 行）

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
    speculative_config: SpeculativeConfig
    quant_config: Optional[QuantizationConfig]
    compilation_config: CompilationConfig
    ...
```

`compute_hash()` 用于 CUDA graph / compile 缓存键。

## ModelConfig（节选）

| 字段 | 含义 |
|------|------|
| `model` | 路径或 HF id |
| `tokenizer` | tokenizer 名 |
| `dtype` | float16/bfloat16/auto |
| `max_model_len` | 最大序列长 |
| `trust_remote_code` | 执行 HF 自定义代码 |
| `runner_type` | generate/pooling |
| `hf_config` | Transformers config 对象 |

## CacheConfig

见 [05-kv-cache-paged-attention.md](./05-kv-cache-paged-attention.md)：

- `block_size`、`gpu_memory_utilization`
- `enable_prefix_caching`、`swap_space`

## SchedulerConfig（节选）

| 字段 | 含义 |
|------|------|
| `max_num_batched_tokens` | 单 step token 上限 |
| `max_num_seqs` | 最大并发 seq |
| `max_model_len` | 与 model 对齐 |
| `enable_chunked_prefill` | 分块 prefill |
| `scheduler_cls` | 可插拔 scheduler 类 |

## ParallelConfig（节选）

| 字段 | 含义 |
|------|------|
| `tensor_parallel_size` | TP |
| `pipeline_parallel_size` | PP |
| `world_size` | 总进程数 |

## SpeculativeConfig

投机解码：

- `model` draft 模型
- `num_speculative_tokens`
- method：eagle、ngram、medusa 等

## QuantizationConfig

`--quantization awq/gptq/fp8/...` 解析为层间量化方法。

## CompilationConfig

`torch.compile` 级别与范围。

## 环境变量

`vllm/envs.py` — 重要项：

| 变量 | 含义 |
|------|------|
| `VLLM_USE_V1` | 0/1 选 V0/V1 |
| `VLLM_ATTENTION_BACKEND` | 强制 attention backend |
| `VLLM_LOGGING_LEVEL` | 日志级别 |
| `VLLM_WORKER_MULTIPROC_METHOD` | spawn/fork |

## EngineArgs / AsyncEngineArgs

`engine/arg_utils.py` — CLI 与 `VllmConfig` 桥梁：

```python
from vllm.engine.arg_utils import AsyncEngineArgs
args = AsyncEngineArgs(model="...", tensor_parallel_size=2)
vllm_config = args.create_engine_config()
```

## 配置优先级

CLI > 环境变量 > 默认值

## 示例：高并发服务

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --tensor-parallel-size 1
```

## 示例：长上下文

```bash
vllm serve model --max-model-len 131072 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.9
```

（需模型与 GPU 显存支持。）

## 阅读源码

搜索 `@dataclass` in `config.py` 得完整字段列表；改配置时注意 `compute_hash()` 注释——影响 graph 的字段需纳入 hash。
