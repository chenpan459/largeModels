# 12 - 快速参考

## 关键路径

| 主题 | 路径 |
|------|------|
| GPT 入口 | `pretrain_gpt.py` |
| 训练主循环 | `megatron/training/training.py` |
| 参数 | `megatron/training/arguments.py` |
| 并行 | `megatron/core/parallel_state.py` |
| TP 层 | `megatron/core/tensor_parallel/layers.py` |
| PP 调度 | `megatron/core/pipeline_parallel/schedules.py` |
| GPT 模型 | `megatron/core/models/gpt/gpt_model.py` |
| 配置 | `megatron/core/transformer/transformer_config.py` |
| MoE | `megatron/core/transformer/moe/` |
| Checkpoint | `megatron/training/checkpointing.py` |
| 数据预处理 | `tools/preprocess_data.py` |

## 并行 CLI

| 参数 | 含义 |
|------|------|
| `--tensor-model-parallel-size` | TP |
| `--pipeline-model-parallel-size` | PP |
| `--context-parallel-size` | CP |
| `--expert-model-parallel-size` | EP |
| `--sequence-parallel` | SP |

## Batch / 训练 CLI

| 参数 | 含义 |
|------|------|
| `--micro-batch-size` | 微批 |
| `--global-batch-size` | 全局 batch |
| `--train-iters` | 迭代数 |
| `--seq-length` | 序列长 |
| `--lr` | 学习率 |
| `--bf16` | BF16 训练 |
| `--use-distributed-optimizer` | 分片 optimizer |

## MoE CLI（节选）

| 参数 | 含义 |
|------|------|
| `--num-experts` | Expert 数 |
| `--moe-router-topk` | Top-K |
| `--moe-token-dispatcher-type` | flex / alltoall |
| `--overlap-moe-expert-parallel-comm` | EP 重叠 |

## 运行模板

```bash
# 安装
uv pip install -e .

# 单机 mock
torchrun --nproc_per_node=1 pretrain_gpt.py --mock-data ...

# 多卡
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  ...

# 预处理
python tools/preprocess_data.py --input ... --output-prefix ...
```

## 版本

- Megatron Core / LM：**0.15.0**（README）
- Python：0.17.0 起要求 ≥3.12（路线图）

## 官方链接

- Docs: https://docs.nvidia.com/megatron-core/developer-guide/latest/
- Parallelism: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- Bridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge
- MoE Roadmap: https://github.com/NVIDIA/Megatron-LM/issues/1729

## 本仓库文档

- `Megatron-LMDoc/README.md` — 索引
- `06-Research/beyond-nanogptDoc/` — 并行原理前置
- `06-Research/README.md` — Research 模块索引
