# 01 - 项目总览

## 定位

**vLLM** = **Easy, fast, and cheap LLM serving**

面向生产的推理引擎，解决 LLM serving 三大瓶颈：

1. **KV Cache 显存浪费** → PagedAttention（块式管理，类似 OS 虚拟内存）
2. **请求到达不均** → Continuous Batching（动态合并 prefill/decode）
3. **算子开销** → FlashAttention、CUDA Graph、量化 kernel

## V0 与 V1 双架构（重要）

当前源码 **默认启用 V1**（`vllm/envs.py` 中 `VLLM_USE_V1: bool = True`），而非文档早期描述的「需手动 `VLLM_USE_V1=1` 开启」。

| | V0（legacy） | V1（当前默认） |
|---|-------------|----------------|
| 路径 | `vllm/engine/`、`vllm/worker/` | `vllm/v1/` |
| Engine | `engine/llm_engine.py` | `v1/engine/llm_engine.py` |
| 内循环 | `LLMEngine.step()` | `EngineCore.step()` |
| Scheduler | `core/scheduler.py` | `v1/core/sched/scheduler.py` |
| Worker | `worker/model_runner.py` | `v1/worker/gpu_model_runner.py` |
| Attention | `attention/backends/` | `v1/attention/backends/` |
| 抢占 | CPU swap 或 recompute | **仅 recompute**（无 CPU swap） |
| 开关 | `VLLM_USE_V1=0` 强制 V0 | 默认即 V1；oracle 可自动回退 |

### 导入时别名（用户无感）

当 `VLLM_USE_V1=True` 时，模块加载阶段会替换公开类：

```python
# engine/llm_engine.py 末尾
from vllm.v1.engine.llm_engine import LLMEngine  # 覆盖 V0 定义

# engine/async_llm_engine.py 末尾
from vllm.v1.engine.async_llm import AsyncLLM as AsyncLLMEngine
```

因此 `from vllm import LLM` 和 `vllm serve` 在默认环境下 **实际走 V1 路径**，无需显式 import `v1` 包。

### 自动选择逻辑

`EngineArgs.create_engine_config()`（`engine/arg_utils.py`）会调用 `_is_v1_supported_oracle()`：

- 若用户未设置 `VLLM_USE_V1`，根据模型/硬件/特性兼容性自动决定
- 不兼容时回退 V0 或报错（取决于 `VLLM_USE_V1` 是否被显式设为 1）

详见 [14-v0-v1-migration.md](./14-v0-v1-migration.md)。

## 目录结构（完整版）

```
vllm/
├── config.py                 # VllmConfig 及全部子配置（3900+ 行）
├── envs.py                   # 环境变量默认值
├── engine/                   # V0 Engine + arg_utils
├── core/                     # V0 Scheduler
├── worker/                   # V0 Worker / ModelRunner / CacheEngine
├── sequence.py               # V0 SequenceGroup / SequenceGroupMetadata
├── attention/
│   ├── layer.py              # 统一 Attention 模块
│   ├── selector.py           # get_attn_backend()
│   └── backends/             # V0 backends（FlashInfer、XFormers 等）
├── model_executor/           # 模型、层、量化（598+ 文件）
│   ├── model_loader/         # 权重加载
│   ├── models/               # 各架构 + registry.py
│   └── layers/               # Linear、MoE、Rotary、quantization/
├── entrypoints/
│   ├── llm.py                # 公开 LLM 类
│   └── openai/               # FastAPI OpenAI 兼容服务
├── platforms/                # cuda.py / rocm.py — backend 与 worker 选择
├── distributed/              # TP/PP group、collective
├── spec_decode/              # V0 投机解码 worker
└── v1/                       # V1 全栈
    ├── engine/               # EngineCore、LLMEngine、AsyncLLM、Processor
    ├── core/
    │   ├── sched/            # Scheduler
    │   ├── kv_cache_manager.py
    │   ├── block_pool.py
    │   └── encoder_cache_manager.py
    ├── worker/               # gpu_worker、gpu_model_runner、block_table
    ├── executor/             # MultiprocExecutor、RayDistributedExecutor
    ├── attention/backends/   # V1 FlashAttention、Triton、MLA
    ├── sample/               # V1 Sampler、RejectionSampler
    ├── spec_decode/          # ngram、eagle proposer
    └── structured_output/    # xgrammar、guidance 约束解码
```

## 核心特性矩阵

| 特性 | V0 | V1 | 说明 |
|------|----|----|------|
| PagedAttention | ✓ | ✓ | block table + FlashAttention |
| Continuous Batching | ✓ | ✓ | 每 step 重调度 |
| Chunked Prefill | 可配置 | **强制开启** | V1 `_set_default_args_v1()` |
| Prefix Caching | ✓ | ✓（默认开） | block hash 复用 |
| Spec Decode | Medusa/MTP/… | **ngram、eagle** | oracle 限制 V1 方法 |
| Structured Output | guided decoding | grammar bitmask | 不同实现 |
| LoRA | ✓ | ✓ | `lora_model_runner_mixin.py` |
| TP | ✓ | ✓ | MultiprocExecutor |
| PP | ✓ | Ray only | V1 需 `--distributed-executor-backend ray` |
| Embedding/Pooling | ✓ | **不支持** | V1 processor 拒绝 |
| CPU KV Swap | ✓ | ✗ | V1 `num_cpu_blocks=0` |

## 入口方式

```bash
# CLI 服务（内部 AsyncLLM + EngineCore 子进程）
vllm serve meta-llama/Llama-3.2-3B --host 0.0.0.0 --port 8000

# Python 同步 API
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.2-3B")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=32))
```

## 官方资源

- 文档：https://docs.vllm.ai
- 论文：[Efficient Memory Management for LLM Serving (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- 上游：https://github.com/vllm-project/vllm
- 许可证：Apache 2.0

## 与本仓库其他模块

| 模块 | 关系 |
|------|------|
| `llama.cpp` | 本地 GGUF + llama-server，轻量单机 |
| `llama.cppDoc` | KV cache、batch 概念对照 |
| `LLaMA-Factory` | 训练 → HF 权重 → vLLM 部署 |
| `Megatron-LM` | 训练 → Bridge → HF → vLLM |
| `kefu-kb` | 当前 llama-server；可换 vLLM 做 chat 后端 |

## 推荐阅读顺序

1. 本文 → [02-architecture.md](./02-architecture.md) → [09-entrypoints-api.md](./09-entrypoints-api.md)
2. 机制深入：[05-kv-cache](./05-kv-cache-paged-attention.md) → [04-scheduler](./04-scheduler-batch.md) → [03-v1-engine](./03-v1-engine.md)
3. V0/V1 差异：[14-v0-v1-migration.md](./14-v0-v1-migration.md)
4. 模型与内核：[06-model-executor](./06-model-executor.md) → [07-attention-backends](./07-attention-backends.md) → [13-sampling](./13-sampling-structured-output.md)
