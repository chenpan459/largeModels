# 14 - V0/V1 迁移与自动选择

## 当前默认

源码 `vllm/envs.py:76`：

```python
VLLM_USE_V1: bool = True
```

**V1 是默认架构**，无需手动设置环境变量。仅当需要 V0 特性或 V1 不兼容时才设 `VLLM_USE_V1=0`。

## 导入时透明别名

用户 API 不变，底层实现被替换：

```python
# vllm/engine/llm_engine.py 末尾
if envs.VLLM_USE_V1:
    from vllm.v1.engine.llm_engine import LLMEngine as LLMEngine

# vllm/engine/async_llm_engine.py 末尾
if envs.VLLM_USE_V1:
    from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine
```

因此：

```python
from vllm import LLM           # → V1 LLMEngine
# vllm serve                   # → V1 AsyncLLM
```

## 自动选择算法

`EngineArgs.create_engine_config()`（`engine/arg_utils.py`）：

```
if VLLM_USE_V1 环境变量未设置:
    supported = _is_v1_supported_oracle(model_config, parallel_config, ...)
    envs.set_vllm_use_v1(supported)
elif VLLM_USE_V1=1 且 oracle 返回 False:
    raise 错误（不静默回退）
elif VLLM_USE_V1=0:
    强制 V0
```

Oracle 还检查 `current_platform.supports_v1(model_config)`。

## V1 Oracle — 不支持特性清单

以下来自 `_is_v1_supported_oracle()`（`arg_utils.py:1354-1527`），启用这些特性时 V1 不可用：

### 加载与格式

| 特性 | 原因 |
|------|------|
| `load_format=tensorizer` | V1 未实现 |
| `load_format=sharded_state` | V1 未实现 |
| GGUF 权重 | V1 未实现 |

### 调度与 Engine

| 特性 | V1 替代/说明 |
|------|-------------|
| `--preemption-mode` | V1 固定 recompute |
| `--scheduling-policy` | V1 固定 FCFS |
| `--num-scheduler-steps` | 不支持 multi-step |
| `--scheduler-delay-factor` | 不支持 |
| `--disable-async-output-proc` | V1 架构不同 |
| `--additional-config` | 不支持 |
| Custom `--logits-processor-pattern` | 用 structured output |

### 模型与任务

| 特性 | 说明 |
|------|------|
| Prompt adapter | V0 only |
| 非 generate runner（embedding/pooling） | V1 processor 拒绝 |
| `is_v1_compatible=False` 模型 | Mamba、encoder-decoder 等 |
| Concurrent partial prefills | V1 不支持 |

### 硬件与 Backend

| 特性 | 说明 |
|------|------|
| GPU SM < 80（非 Ampere+） | V1 要求 |
| FP8 KV cache | 需 FA FP8 支持 |
| `VLLM_ATTENTION_BACKEND=FLASHINFER` | V1 attention 不支持 FlashInfer |
| Attention backend 不在 V1 列表 | 见下方 V1 backend 列表 |

### 并行

| 特性 | 说明 |
|------|------|
| PP + 非 Ray backend | V1 PP 必须 Ray |
| KV transfer / disaggregated prefill | V0 only |

### 解码与观测

| 特性 | 说明 |
|------|------|
| Guided decoding 非 xgrammar/guidance/auto | V1 限定 backend |
| OTLP tracing | V1 不支持 |
| Spec decode 非 ngram/eagle | Medusa/MTP 等需 V0 |

### V1 允许的 Attention Backend

```
FLASH_ATTN_VLLM_V1, FLASH_ATTN, PALLAS, PALLAS_VLLM_V1,
TRITON_ATTN_VLLM_V1, TRITON_MLA, FLASHMLA
```

## V1 默认参数覆盖

`_set_default_args_v1()`（`arg_utils.py:1634-1693`）在用户未指定时强制/设置：

| 参数 | V1 行为 |
|------|---------|
| `enable_chunked_prefill` | **强制 True** |
| `enable_prefix_caching` | 默认 True |
| `prefix_caching_hash_algo` | `builtin` |
| `scheduler_cls` | `vllm.v1.core.sched.scheduler.Scheduler` |
| `max_num_batched_tokens` | H100: 8192；其他: 2048 |
| `max_num_seqs` | H100: 1024；其他: 256 |

## 架构对照表

| 组件 | V0 | V1 |
|------|----|----|
| Engine | `engine/llm_engine.py` | `v1/engine/llm_engine.py` |
| 内循环 | `LLMEngine.step()` | `EngineCore.step()` |
| Scheduler | `core/scheduler.py` | `v1/core/sched/scheduler.py` |
| 调度模型 | prefill/decode 阶段 | 统一 token 追赶 |
| Request | `SequenceGroup` | `Request` |
| 调度输出 | `SequenceGroupMetadata` | `SchedulerOutput` |
| Worker | `worker/model_runner.py` | `v1/worker/gpu_model_runner.py` |
| KV swap | CPU swap 支持 | **仅 recompute** |
| Attention | `attention/backends/` | `v1/attention/backends/` |
| Sampling | `model_executor/layers/sampler.py` | `v1/sample/sampler.py` |
| Spec decode | `spec_decode/` worker | `v1/spec_decode/` |
| Structured output | guided decoding | grammar bitmask |
| OpenAI client | MQ 或 in-proc | 始终 EngineCoreProc |
| Embedding API | 支持 | **不支持** |

## 抢占行为差异（重要）

| | V0 | V1 |
|---|----|----|
| 模式 | `PreemptionMode.SWAP` 或 `RECOMPUTE` | 仅 RECOMPUTE |
| CPU swap | `blocks_to_swap_in/out` | `num_cpu_blocks=0` |
| 恢复 | Swap 回 GPU 或重算 | 完整 prefill 重算 |
| 配置 | `--swap-space`、`--preemption-mode` | 无效 |

迁移时注意：V1 抢占开销可能更高（无 swap），但架构更简单。

## 何时用 V0

- 需要 embedding/pooling/rerank API（V1 不支持）
- 需要 Medusa/MTP spec decode
- 需要 GGUF 直接加载
- 需要 prompt adapter
- 需要自定义 scheduling policy 或 logits processor
- 模型 `is_v1_compatible=False`
- GPU 较老（SM < 80）

```bash
VLLM_USE_V1=0 vllm serve model
```

## 何时用 V1（默认）

- 生产 text generation 高并发
- 需要更低 overhead prefix caching
- 需要 xgrammar structured output
- NVIDIA Ampere+ GPU
- 标准 HF 权重

```bash
# 默认即 V1，无需设置
vllm serve model
```

## 调试 V1 兼容性

```bash
# 查看 oracle 决策
VLLM_LOGGING_LEVEL=DEBUG vllm serve model 2>&1 | grep -i v1

# 强制 V1 看报错
VLLM_USE_V1=1 vllm serve model

# 同进程调试
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=1 python -c "
from vllm import LLM
llm = LLM(model='...', enforce_eager=True)
"
```

## 实验性 V1 特性

Oracle 中标记 experimental、默认 off：

- ngram speculative decoding
- eagle speculative decoding
- Pipeline parallelism（需 Ray）

启用方式见官方 docs 对应 `--speculative-model` / `--speculative-method` 参数。

## 迁移检查清单

- [ ] 模型在 V1 registry 中且 `is_v1_compatible=True`
- [ ] 不使用 embedding/pooling API
- [ ] 不使用自定义 logits_processor
- [ ] 不使用 GGUF/tensorizer 加载
- [ ] GPU SM ≥ 80
- [ ] 如用 PP → `--distributed-executor-backend ray`
- [ ] 如用 spec decode → 仅 ngram/eagle
- [ ] 理解 preempt 变为 full recompute
- [ ] 测试 structured output backend（xgrammar/guidance）

## 关键源码

| 主题 | 文件:行 |
|------|---------|
| V1 默认 | `envs.py:76` |
| Engine 别名 | `engine/llm_engine.py:2169-2171` |
| Oracle | `engine/arg_utils.py:1354-1527` |
| V1 默认参数 | `engine/arg_utils.py:1634-1693` |
| V0 抢占 | `core/scheduler.py` PreemptionMode |
| V1 抢占 | `v1/core/sched/scheduler.py:201-217` |
