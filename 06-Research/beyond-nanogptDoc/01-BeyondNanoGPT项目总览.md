# 01 - 项目总览

## 一句话定位

**Beyond-NanoGPT** = 在 nanoGPT 最小 GPT 之上，用 **手写 PyTorch** 系统实现从 Attention 变体、MoE/Mamba、扩散模型、深度 RL 到 GPU 通信与 Agent 的 **~100 项现代技术**，代码即教程。

## 设计哲学

| 原则 | 体现 |
|------|------|
| 手写优先 | 多数模块仅用 `nn.Linear` 等基础算子，不用高层封装 |
| 注释即讲义 | 文件头与行内注释解释论文与实现细节 |
| 单 GPU 可跑 | 除 DDP/TP 外，脚本默认单卡；多卡用 `torchrun` |
| 可读 > 简洁 | 故意不抽象，便于对照论文 re-implement |
| CLI 统一 | 各 `.py` 底部 `argparse`，支持 `--verbose`、`--wandb` |

## 目录结构

```
beyond-nanogpt/
├── README.md              # 完整路线图（checkbox）
├── LESSONS.md             # 作者工程心得（>10k 行 PyTorch 总结）
├── language-models/       # 12 文件：核心 Transformer + 3 版 DataLoader + 5 Notebook
├── attention-variants/    # 7 个 .ipynb
├── architectures/         # 8 个 train_*.py
├── generative-models/     # 8 个训练/引导脚本
├── rl/
│   ├── fundamentals/      # DQN, REINFORCE, PPO
│   ├── actor-critic/        # A2C, A3C, IMPALA*, DDPG, SAC*
│   ├── model-based/         # MPC, Expert Iteration (MCTS)
│   ├── llms/                # GRPO (GSM8K, humor)
│   └── chess/               # AlphaZero 进行中
├── mlsys/
│   ├── comms.py             # scatter/gather/ring&tree allreduce
│   ├── train_ddp.py         # 手写 DDP
│   ├── train_tp.py          # Tensor Parallel*
│   └── kernels/             # 7 个 Triton 内核
├── evals/                   # GSM8K, MMLU, SimpleQA
├── rag/                     # intro_rag.py
└── agents/
    ├── basic-search-use/    # 搜索增强 QA
    └── coding-agent/        # ReAct + 工具 + 记忆
```

**规模**：约 97 个源文件（不含 `.git`），无 monorepo 式子包，按 **topic 目录** 组织。

## 技术覆盖清单（按 README 路线图）

### 已实现（[x]）

| 类别 | 数量 | 代表文件 |
|------|------|----------|
| Architectures | 8 | `train_moe.py`, `train_mamba.py`, `train_dit.py` |
| Attention | 7 notebooks | `gqa.ipynb`, `mla.ipynb`, `linear_attention.ipynb` |
| Language Models | 6+ | `transformer.py`, `dataloaders/`, `train_mtp.py` |
| RL (classic) | 10+ | `train_ppo.py`, `train_sac.py`, `train_impala.py` |
| RL (LLM) | 2 | `train_grpo_gsm.py`, `train_grpo_humor.py` |
| Generative | 8 | `train_ddpm.py`, `train_flow_matching.py` |
| MLSys | 10 | `comms.py`, `train_ddp.py`, `kernels/tiled_gemm.py` |
| Evals | 3 | `eval_gsm8k.py`, `eval_mmlu.py`, `eval_simpleqa.py` |
| RAG | 1 | `intro_rag.py` |
| Agents | 2 子项目 | `coding-agent/`, `basic-search-use/` |
| Chess | 部分 | `MCTS.py`, `model.py`, `env.py`（训练环未完成） |

### 未完成（[ ]）节选

- RLHF / DPO on UltraFeedback
- PETS、Chess 完整训练与 Elo 评估
- Ring Attention、Paged Attention、Continuous Batching
- FlashAttention Forward（Triton）
- NeRF、Graph RAG、Multi-Agent Research

详见 [11-路线图与实现缺口.md](./11-路线图与实现缺口.md)。

## 依赖

```bash
pip install torch numpy torchvision wandb tqdm transformers datasets \
  diffusers matplotlib pillow jupyter gym
```

**按需额外安装**：

| 场景 | 包 |
|------|-----|
| Chess | `python-chess`, `pytest` |
| Coding Agent | `together`, `anthropic`, `pydantic` |
| RAG eval | `sentence-transformers`, API keys |
| Triton kernels | `triton` |

## 运行方式

### Python 脚本

```bash
cd architectures
python train_moe.py --verbose --wandb
```

参数定义在 **各文件底部** `if __name__ == "__main__"` 的 `argparse`。

### Jupyter Notebook

Attention 变体、BPE、RoPE、KV Cache、投机解码等适合 **逐步执行**：

```bash
jupyter notebook attention-variants/gqa.ipynb
```

### 分布式

```bash
torchrun --nproc_per_node=2 mlsys/train_ddp.py
torchrun --nproc_per_node=2 mlsys/train_tp.py
```

## 与 nanoGPT 的差异

| 维度 | nanoGPT | Beyond-NanoGPT |
|------|---------|----------------|
| 目标 | 最小可训练 GPT | 广度：多架构/多范式 |
| 代码量 | ~300 行核心 | 单文件可达数百行 + 长注释 |
| 数据 | shakespeare / openwebtext | TinyStories、MNIST、CartPole、GSM8K 等 |
| 抽象 | 一个 `GPT` 类 | 每篇论文一个目录/文件 |
| 工程 | 极简 | DataLoader v0→v2、手写 DDP、Triton |

## 代码风格特征

1. **文件头 docstring**：论文链接 + 实现要点 + 与相关文件关系
2. **dataclass Config**：`transformer.py` 中 `AttentionConfig`、`TransformerConfig` 等
3. **ACT2FN 字典**：统一激活函数映射
4. **device 默认 cuda**：`torch.device("cuda" if torch.cuda.is_available() else "cpu")`
5. **wandb 可选**：`--wandb` 开关，便于实验追踪

## Citation

```bibtex
@misc{kumar2025beyond,
  author = {Tanishq Kumar},
  title = {Beyond-NanoGPT: From LLM Beginner to AI Researcher},
  year = {2025},
  howpublished = {\url{https://github.com/tanishqkumar/beyond-nanogpt}}
}
```
