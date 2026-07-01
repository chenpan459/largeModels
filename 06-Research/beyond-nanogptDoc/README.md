# Beyond-NanoGPT 项目文档

本目录包含对 `/home/cp/work2/largeModels/06-Research/beyond-nanogpt` 项目的结构化源码分析文档。

## 项目定位

**Beyond-NanoGPT** 是连接 [nanoGPT](https://github.com/karpathy/nanoGPT) 与前沿 AI 研究的 **教育型代码库**：约 **97 个文件**、**~100 项现代深度学习技术** 的手写 PyTorch 实现，注释详尽，单 GPU 可运行。

作者：Tanishq Kumar · 许可证：见源码 `LICENSE` · 上游：https://github.com/tanishqkumar/beyond-nanogpt

## 文档索引

### 概览与路径

| 文档 | 说明 |
|------|------|
| [01-project-overview.md](./01-project-overview.md) | 目录结构、技术清单、依赖与运行方式 |
| [02-learning-path.md](./02-learning-path.md) | 从 nanoGPT 到研究的模块关系与学习顺序 |

### 核心专题

| 文档 | 说明 |
|------|------|
| [03-language-models.md](./03-language-models.md) | Transformer、DataLoader、KV Cache、投机解码 |
| [04-attention-variants.md](./04-attention-variants.md) | MHSA、GQA、Linear/MLA/Sparse Attention |
| [05-architectures.md](./05-architectures.md) | ViT、MoE、Mamba、DiT、ResNet 等 |
| [06-generative-models.md](./06-generative-models.md) | GAN、VAE、DDPM、Flow Matching |
| [07-reinforcement-learning.md](./07-reinforcement-learning.md) | DQN→PPO→GRPO、Chess/AlphaZero |
| [08-mlsys.md](./08-mlsys.md) | GPU 通信、DDP、TP、Triton Kernels |
| [09-evals-rag.md](./09-evals-rag.md) | GSM8K、MMLU、SimpleQA、intro RAG |
| [10-agents.md](./10-agents.md) | 搜索 Agent、Coding Agent（ReAct + Tools） |

### 实践参考

| 文档 | 说明 |
|------|------|
| [11-roadmap-and-gaps.md](./11-roadmap-and-gaps.md) | 已实现 vs 路线图 TODO |
| [12-engineering-lessons.md](./12-engineering-lessons.md) | LESSONS.md 精华与工程模式 |
| [13-quick-reference.md](./13-quick-reference.md) | 命令行、参数、文件速查 |

## 项目路径

```
/home/cp/work2/largeModels/06-Research/beyond-nanogpt/
├── language-models/       # LLM 核心 + DataLoader + Notebooks
├── attention-variants/    # 7 个 Attention Jupyter
├── architectures/         # ViT / MoE / Mamba / DiT …
├── generative-models/     # GAN / VAE / DDPM / Flow
├── rl/                    # 经典 RL + LLM GRPO + Chess
├── mlsys/                 # comms / DDP / TP / kernels
├── evals/                 # GSM8K / MMLU / SimpleQA
├── rag/                   # RAG 入门
└── agents/                # 搜索 + Coding Agent
```

## 推荐阅读顺序

1. **巩固 nanoGPT 之后**：01 → 02 → 03（Transformer + dataloader）
2. **LLM 推理优化**：03（KV_cache、speculative_decoding）→ 04（GQA、MLA）
3. **前沿架构**：05（MoE*、Mamba*）
4. **对齐与 Agent**：07（GRPO）→ 10（Coding Agent）
5. **分布式训练**：08 → 本仓库 `Megatron-LM/`
6. **工程心法**：12-engineering-lessons

带 * 标记的实现作者标注为 particularly tricky。

## 快速开始

```bash
cd /home/cp/work2/largeModels/06-Research/beyond-nanogpt

pip install torch numpy torchvision wandb tqdm transformers datasets \
  diffusers matplotlib pillow jupyter gym

# 示例：训练 DiT
cd architectures && python train_dit.py

# 示例：PPO + wandb
cd rl/fundamentals && python train_ppo.py --verbose --wandb
```

## 与本仓库其他模块的关系

| 模块 | 关系 |
|------|------|
| `01-模型原理/nanoGPT` | 前置：最小 GPT 训练循环 |
| `02-训练/LLaMA-Factory` | 工业级微调（beyond-nanogpt 偏原理） |
| `03-推理部署/llama.cpp` | KV Cache、投机解码概念对照 |
| `05-RAG/llama_index` | RAG 框架 vs `rag/intro_rag.py` 最小实现 |
| `06-Research/Megatron-LM` | 大规模分布式训练的工业实现 |
| `07-业务应用/kefu-kb` | 业务 RAG MVP |

## 上游项目

- 仓库: https://github.com/tanishqkumar/beyond-nanogpt
- 作者笔记: 源码根目录 `LESSONS.md`
