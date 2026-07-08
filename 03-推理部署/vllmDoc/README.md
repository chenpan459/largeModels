# vLLM 项目文档

本目录包含对 `/home/cp/work2/largeModels/03-推理部署/vllm` 项目的结构化源码分析文档（深度扩展版）。

## 项目定位

**vLLM** 是 UC Berkeley Sky Computing Lab 发起的 **高吞吐 LLM 推理与服务框架**，核心创新 **PagedAttention** + **Continuous Batching**，广泛用于生产 API 服务。

- 论文：[Efficient Memory Management for LLM Serving (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- 上游：https://github.com/vllm-project/vllm
- 许可证：Apache 2.0
- **当前源码默认 V1 架构**（`VLLM_USE_V1=True`）

## 文档索引

### 概览与架构

| 文档 | 说明 |
|------|------|
| [01-project-overview.md](./01-project-overview.md) | 目录结构、V0/V1、特性矩阵、导入别名 |
| [02-architecture.md](./02-architecture.md) | 端到端数据流、EngineCoreClient、统一 token 调度 |
| [14-v0-v1-migration.md](./14-v0-v1-migration.md) | **V0/V1 自动选择、oracle 清单、迁移指南** |

### 核心模块

| 文档 | 说明 |
|------|------|
| [03-v1-engine.md](./03-v1-engine.md) | EngineCore、LLMEngine、Executor、IPC |
| [04-scheduler-batch.md](./04-scheduler-batch.md) | Scheduler、Continuous Batching、抢占（recompute） |
| [05-kv-cache-paged-attention.md](./05-kv-cache-paged-attention.md) | BlockPool、KVCacheManager、Prefix Cache |
| [06-model-executor.md](./06-model-executor.md) | 模型加载、GPUModelRunner、persistent batch |
| [07-attention-backends.md](./07-attention-backends.md) | abstract 层、FlashAttention/MLA、选择逻辑 |
| [08-parallel-distributed.md](./08-parallel-distributed.md) | TP、PP（Ray）、DP、MoE EP |
| [09-entrypoints-api.md](./09-entrypoints-api.md) | OpenAI API 全路由、EngineClient |
| [13-sampling-structured-output.md](./13-sampling-structured-output.md) | **Sampler、Spec Decode、Grammar Bitmask** |

### 实践与参考

| 文档 | 说明 |
|------|------|
| [10-config-reference.md](./10-config-reference.md) | VllmConfig、CompilationLevel、环境变量 |
| [15-quantization-catalog.md](./15-quantization-catalog.md) | **GPTQ/AWQ/FP8/Marlin 量化目录** |
| [11-llama-cpp-integration.md](./11-llama-cpp-integration.md) | 与 llama.cpp / kefu-kb 对照 |
| [12-quick-reference.md](./12-quick-reference.md) | 命令、环境变量、源码索引 |

## 项目路径

```
/home/cp/work2/largeModels/03-推理部署/vllm/
├── vllm/
│   ├── v1/                 # V1 架构（默认）
│   │   ├── engine/         # EngineCore、Processor
│   │   ├── core/sched/     # Scheduler
│   │   ├── worker/         # GPUModelRunner
│   │   ├── sample/         # V1 Sampler
│   │   └── attention/      # V1 Backends
│   ├── engine/             # V0 Engine + arg_utils（含 V1 oracle）
│   ├── worker/             # V0 Worker
│   ├── model_executor/     # 模型、层、quantization/
│   ├── entrypoints/openai/ # API Server
│   ├── attention/          # Attention 抽象 + V0 backends
│   └── config.py           # VllmConfig
└── tests/
```

## 推荐阅读顺序

1. **入门**：01 → 02 → 09（跑 OpenAI API）
2. **V0/V1 理解**：14 → 01
3. **核心机制**：05 → 04 → 03
4. **执行路径**：06 → 13 → 07
5. **分布式与配置**：08 → 10 → 15
6. **对照 llama.cpp**：11 → `llama.cppDoc/16-kv-cache-memory.md`

## 快速开始

```bash
pip install vllm

# OpenAI 兼容服务（默认 V1）
vllm serve meta-llama/Llama-3.2-3B --host 0.0.0.0 --port 8000

# Python API
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.2-3B")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=32))

# 强制 V0（如需 embedding API 等）
VLLM_USE_V1=0 vllm serve model
```

## 与本仓库其他模块

| 模块 | 关系 |
|------|------|
| `llama.cpp` | 本地 GGUF + llama-server |
| `llama.cppDoc` | KV cache、batch 概念对照 |
| `LLaMA-Factory` | 训练 → HF 权重 → vLLM |
| `Megatron-LM` | 训练 → Bridge → vLLM 部署 |
| `kefu-kb` | 当前 llama-server chat；可换 vLLM |

## 文档版本说明

本文档基于本地 vLLM 源码分析编写，覆盖 V1 默认架构、EngineCoreClient IPC、统一 token 调度、V1 recompute 抢占、grammar bitmask structured output、完整 API 路由表、量化目录等。若 upstream 升级，请以 `vllm/` 源码为准。
