# Megatron-LM 项目文档

本目录包含对 `/home/cp/work2/largeModels/06-Research/Megatron-LM` 项目的结构化源码分析文档。

## 项目定位

**Megatron-LM** + **Megatron Core** 是 NVIDIA 维护的 **大规模 Transformer 训练框架**：

- **Megatron Core**：可组合 GPU 优化组件库（并行、MoE、FP8、模型块）
- **Megatron-LM**：参考训练脚本 + 配置 + 工具（`pretrain_gpt.py` 等）
- 版本：**0.15.0**（README）· 上游：https://github.com/NVIDIA/Megatron-LM

## 文档索引

### 概览与架构

| 文档 | 说明 |
|------|------|
| [01-Megatron-LM项目总览.md](./01-Megatron-LM项目总览.md) | 双组件结构、目录、版本 |
| [02-架构与训练数据流.md](./02-架构与训练数据流.md) | 分层架构与训练数据流 |

### 核心模块

| 文档 | 说明 |
|------|------|
| [03-并行策略.md](./03-并行策略.md) | TP / PP / DP / EP / CP 五维并行 |
| [04-Megatron-Core模块.md](./04-Megatron-Core模块.md) | Core 模块与 GPTModel |
| [05-Transformer与MoE高级架构.md](./05-Transformer与MoE高级架构.md) | Transformer 层、MoE、MLA、MTP |
| [06-训练循环.md](./06-训练循环.md) | pretrain / train_step / 调度 |
| [07-数据与Checkpoint.md](./07-数据与Checkpoint.md) | 数据集、分布式 checkpoint |
| [08-推理与Megatron-RL.md](./08-推理与Megatron-RL.md) | 推理、Megatron-RL |
| [09-pretrain_gpt.py精读.md](./09-pretrain_gpt.py精读.md) | pretrain_gpt.py 精读 |

### 实践与参考

| 文档 | 说明 |
|------|------|
| [10-工具示例与测试.md](./10-工具示例与测试.md) | tools/、examples/、测试 |
| [11-与本仓库模块对照.md](./11-与本仓库模块对照.md) | 与 beyond-nanogpt / Bridge 对照 |
| [12-快速参考.md](./12-快速参考.md) | CLI 参数与路径速查 |

## 项目路径

```
/home/cp/work2/largeModels/06-Research/Megatron-LM/
├── megatron/
│   ├── core/           # Megatron Core（~555 文件）
│   ├── training/       # 训练循环、参数、checkpoint
│   ├── rl/             # Megatron-RL（部分对外）
│   └── post_training/  # 量化、蒸馏等
├── pretrain_gpt.py     # GPT 预训练/SFT 入口
├── examples/           # 各模型示例脚本
├── tools/              # 数据预处理、checkpoint 转换
└── tests/              # unit + functional tests
```

## 推荐阅读顺序

1. **入门**：01 → 02 → 09（pretrain_gpt）
2. **并行**：03 → 06
3. **模型**：04 → 05
4. **生产**：07 → 10
5. **衔接**：11 → `beyond-nanogptDoc/08-ML系统.md` → Megatron Bridge

## 快速开始

```bash
cd /home/cp/work2/largeModels/06-Research/Megatron-LM
uv pip install -e .

# 单机多卡 GPT 预训练（示例，需准备数据）
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 4 \
  ...
```

官方文档：https://docs.nvidia.com/megatron-core/developer-guide/latest/

## 与本仓库其他模块

| 模块 | 关系 |
|------|------|
| `beyond-nanogpt/mlsys/` | 手写 DDP/TP/comms 教学版 |
| `02-训练/LLaMA-Factory` | 微调框架，非大规模预训练 |
| `03-推理部署/vllm` | 推理 serving，可加载 Megatron 转换权重 |
| Megatron Bridge | HF ↔ Megatron checkpoint 互转（外部仓库） |

## 上游项目

- 仓库: https://github.com/NVIDIA/Megatron-LM
- 许可证: Apache 2.0
- MoE 路线图: https://github.com/NVIDIA/Megatron-LM/issues/1729
