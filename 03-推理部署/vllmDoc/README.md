# vLLM 项目文档

本目录包含对 `/home/cp/work2/largeModels/03-推理部署/vllm` 项目的结构化源码分析文档。

## 项目定位

**vLLM** 是 UC Berkeley Sky Computing Lab 发起的 **高吞吐 LLM 推理与服务框架**，核心创新 **PagedAttention** + **Continuous Batching**，广泛用于生产 API 服务。

- 论文：[Efficient Memory Management for LLM Serving (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- 上游：https://github.com/vllm-project/vllm
- 许可证：Apache 2.0

## 文档索引

### 概览与架构

| 文档 | 说明 |
|------|------|
| [01-project-overview.md](./01-project-overview.md) | 目录结构、V0/V1、特性清单 |
| [02-architecture.md](./02-architecture.md) | 端到端推理数据流 |

### 核心模块

| 文档 | 说明 |
|------|------|
| [03-v1-engine.md](./03-v1-engine.md) | EngineCore、LLMEngine、Executor |
| [04-scheduler-batch.md](./04-scheduler-batch.md) | Scheduler、Continuous Batching |
| [05-kv-cache-paged-attention.md](./05-kv-cache-paged-attention.md) | PagedAttention、BlockPool、Prefix Cache |
| [06-model-executor.md](./06-model-executor.md) | 模型加载、GPUModelRunner |
| [07-attention-backends.md](./07-attention-backends.md) | FlashAttention、MLA、Backend 选择 |
| [08-parallel-distributed.md](./08-parallel-distributed.md) | TP、PP、Ray、Multiproc |
| [09-entrypoints-api.md](./09-entrypoints-api.md) | OpenAI API Server、CLI |

### 实践与参考

| 文档 | 说明 |
|------|------|
| [10-config-reference.md](./10-config-reference.md) | VllmConfig 与子配置 |
| [11-llama-cpp-integration.md](./11-llama-cpp-integration.md) | 与 llama.cpp / kefu-kb 对照 |
| [12-quick-reference.md](./12-quick-reference.md) | 命令与环境变量速查 |

## 项目路径

```
/home/cp/work2/largeModels/03-推理部署/vllm/
├── vllm/
│   ├── v1/                 # V1 架构（默认演进方向）
│   ├── engine/             # V0 Engine
│   ├── worker/             # V0 Worker / ModelRunner
│   ├── model_executor/     # 模型实现与量化层
│   ├── entrypoints/        # API Server、CLI
│   ├── attention/          # Attention 抽象
│   └── config.py           # 统一配置
└── tests/
```

## 推荐阅读顺序

1. **入门**：01 → 02 → 09（跑 OpenAI API）
2. **核心机制**：05 → 04 → 03
3. **模型与内核**：06 → 07
4. **分布式**：08
5. **对照 llama.cpp**：11 → `llama.cppDoc/16-kv-cache-memory.md`

## 快速开始

```bash
pip install vllm

# OpenAI 兼容服务
vllm serve meta-llama/Llama-3.2-3B --host 0.0.0.0 --port 8000

# Python API
from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.2-3B")
outputs = llm.generate(["Hello"], sampling_params=...)
```

## 与本仓库其他模块

| 模块 | 关系 |
|------|------|
| `llama.cpp` | 本地 GGUF + llama-server |
| `llama.cppDoc` | KV cache、batch 概念对照 |
| `Megatron-LM` | 训练 → Bridge → vLLM 部署 |
| `kefu-kb` | 当前用 llama-server，可换 vLLM |
