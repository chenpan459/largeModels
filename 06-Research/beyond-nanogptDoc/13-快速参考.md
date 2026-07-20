# 13 - 快速参考

## 目录 → 入口命令

| 目录 | 示例命令 |
|------|----------|
| `architectures/` | `python train_moe.py --verbose --wandb` |
| `language-models/` | `python train_full.py --verbose` |
| `generative-models/` | `python train_ddpm.py --verbose --wandb` |
| `rl/fundamentals/` | `python train_ppo.py --verbose --wandb` |
| `rl/llms/` | `python train_grpo_gsm.py --verbose --wandb` |
| `mlsys/` | `torchrun --nproc_per_node=2 train_ddp.py` |
| `evals/` | `python eval_gsm8k.py` |
| `rag/` | `python intro_rag.py` |
| `agents/coding-agent/` | `python agent.py --verbose` |
| `rl/chess/` | `pytest test_all.py -v` |

## 通用 CLI 参数

多数训练脚本支持：

| 参数 | 作用 |
|------|------|
| `--verbose` | 打印详细日志 |
| `--wandb` | Weights & Biases 记录 |

具体超参见各文件底部 `argparse` 定义。

## 依赖速查

```bash
# 基础
pip install torch numpy torchvision wandb tqdm transformers datasets \
  diffusers matplotlib pillow jupyter gym

# Chess
pip install python-chess pytest

# Coding Agent
pip install together anthropic pydantic

# Triton kernels
pip install triton

# RAG eval
pip install sentence-transformers
```

## 关键源文件路径

| 主题 | 路径 |
|------|------|
| Transformer 核心 | `language-models/transformer.py` |
| DataLoader v2 | `language-models/dataloaders/dataloader2.py` |
| MoE | `architectures/train_moe.py` |
| Mamba | `architectures/train_mamba.py` |
| PPO | `rl/fundamentals/train_ppo.py` |
| GRPO | `rl/llms/train_grpo_gsm.py` |
| Allreduce | `mlsys/comms.py` |
| DDP | `mlsys/train_ddp.py` |
| Agent | `agents/coding-agent/agent.py` |
| MCTS | `rl/chess/MCTS.py` |

## Notebooks

```bash
jupyter notebook attention-variants/gqa.ipynb
jupyter notebook language-models/KV_cache.ipynb
jupyter notebook language-models/speculative_decoding.ipynb
```

## 环境变量（Agents / RAG）

```bash
export TOGETHER_API_KEY=...
export ANTHROPIC_API_KEY=...      # optional
export GOOGLE_SEARCH_KEY=...      # search tool
export SEARCH_ENGINE_ID=...
export GITHUB_PAT=...             # coding-agent PR
export WANDB_API_KEY=...          # --wandb
```

## 标记 * 的 tricky 实现

| 文件 | 难点 |
|------|------|
| `train_moe.py` | expert scatter/gather 向量化 |
| `train_mamba.py` | SSM 离散化与 selective params |
| `linear_attention.ipynb` | 核技巧避免 O(n²) |
| `mla.ipynb` | 低秩 KV 压缩 |
| `train_impala.py` | 分布式 V-trace + producer-consumer |
| `train_sac.py` | 双 Q + 熵 |
| `train_ddpm.py` | U-Net + 噪声 schedule |
| `comms.py` | ring allreduce deadlock |
| `train_tp.py` | 列/行并行切分 |

## 与本仓库交叉索引

| 需求 | 文档 |
|------|------|
| nanoGPT 前置 | `01-模型原理/nanoGPT/` |
| 推理/KV | `03-推理部署/llama.cppDoc/` |
| 量化 kernel | `04-量化内核/ggmlDoc/` |
| RAG 框架 | `05-RAG/llama_indexDoc/` |
| 分布式预训练 | `06-Research/Megatron-LM/` |
| 客服 RAG | `07-业务应用/kefu-kb/` |

## 上游

- GitHub: https://github.com/tanishqkumar/beyond-nanogpt
- 作者: Tanishq Kumar (tanishq@stanford.edu)
- 笔记: `LESSONS.md`
