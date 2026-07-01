# 01 - 项目总览

## 定位

**vLLM** = **Easy, fast, and cheap LLM serving**

面向生产的推理引擎，解决 LLM serving 三大瓶颈：

1. **KV Cache 显存浪费** → PagedAttention（块式管理，类似 OS 虚拟内存）
2. **请求到达不均** → Continuous Batching（动态合并 prefill/decode）
3. **算子开销** → FlashAttention、CUDA Graph、量化 kernel

## V0 与 V1 双架构

2025 起 **V1** 为重大架构升级（官方称 ~1.7× 加速）：

| | V0（legacy） | V1（新默认方向） |
|---|-------------|------------------|
| 路径 | `vllm/engine/`、`vllm/worker/` | `vllm/v1/` |
| Engine | `LLMEngine` | `v1/engine/llm_engine.py` |
| Scheduler | legacy scheduler | `v1/core/sched/scheduler.py` |
| Worker | `worker/model_runner.py` | `v1/worker/gpu_model_runner.py` |
| 开关 | `VLLM_USE_V1=0` | `VLLM_USE_V1=1` |

## 目录结构

```
vllm/
├── config.py              # VllmConfig（3900+ 行）
├── attention/             # Attention 层 + backend
├── model_executor/        # 模型、层、量化（598+ 文件）
├── entrypoints/openai/    # OpenAI API
├── engine/                # V0 Engine
├── worker/                # V0 Worker
└── v1/                    # V1 全栈
    ├── engine/            # EngineCore
    ├── core/sched/        # Scheduler
    ├── core/              # KVCacheManager、BlockPool
    ├── worker/            # GPUModelRunner
    └── executor/          # MultiprocExecutor
```

## 核心特性

PagedAttention、Continuous Batching、Prefix Caching、Chunked Prefill、Speculative Decoding（EAGLE/Medusa）、GPTQ/AWQ/FP8、TP/PP、OpenAI API、Multi-LoRA。

## 入口

```bash
vllm serve meta-llama/Llama-3.2-3B
from vllm import LLM
```

## 官方

- https://docs.vllm.ai
- 论文：https://arxiv.org/abs/2309.06180
