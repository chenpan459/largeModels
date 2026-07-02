# 第 1 章 — 环境与工具

## 开始之前你需要了解的内容

### 「Python 到底是什么？」

Python 只是一种**用来告诉计算机该做什么的语言**。你在 `.py` 文件里写指令，Python 会逐条「阅读」并执行它们。如果你以前写过 Python——哪怕只是 `print("hello")`——你就已经准备好了。

### 「什么是 GPU，为什么需要它？」

**类比：** 想象你需要给 10,000 块小瓷砖上色。

- **CPU** 就像一位大师级画家，一次只画一块瓷砖——精确但慢。
- **GPU** 就像 10,000 个艺术生，每人同时画一块瓷砖——更快，尽管每个学生都比大师「笨」一些。

训练神经网络涉及数百万次**相同、独立的数学运算**（矩阵乘法）。GPU 拥有数千个专为这类任务设计的小核心。训练时 GPU 可能比 CPU 快 50–100 倍。

**你一定需要 GPU 吗？** 不一定——我们的小型测试模型可以在 CPU 上运行，只是非常慢（分钟级 vs 小时级）。真正训练时，GPU 必不可少。

| 你的硬件 | 能训练什么 | 大致速度 |
|---|---|---|
| 仅 CPU | 微型模型（4 层，256 维） | 数小时 |
| Apple M1/M2/M3 | 小模型（12 层，768 维） | 数小时 |
| RTX 3060/4060（12GB） | GPT-2 small（1.24 亿参数） | 数小时 |
| RTX 3090/4090（24GB） | GPT-2 medium（3.5 亿） | 数小时 |
| A100（80GB） | GPT-2 large（7.74 亿） | 数小时 |

### 「什么是虚拟环境？」

虚拟环境（`venv`）就像为这个项目准备的**干净、空的厨房**。没有它，你会把本项目的「食材」（Python 包）和电脑上其他所有东西混在一起——当两个项目需要同一包的不同版本时就会冲突。

```bash
# 创建一个干净的厨房
python -m venv gpt_env

# 进入虚拟环境
source gpt_env/bin/activate          # Mac/Linux
# 或者：
gpt_env\Scripts\activate             # Windows

# 现在 pip install 只影响这个厨房
# 退出：输入 `deactivate`
```

### 「什么是 pip？」

`pip` 是 Python 的**包安装器**。它从互联网下载别人写好的代码（库）并安装到你的环境中。可以把它想成 Python 代码的「应用商店」。

### 「什么是 PyTorch？」

PyTorch 是我们用来构建神经网络的框架。它提供：

| PyTorch 特性 | 作用 | 类比 |
|---|---|---|
| `torch.Tensor` | 多维数组 | 类似 NumPy 数组，但可以放在 GPU 上 |
| `torch.nn.Module` | 网络的构建块 | 可以拼在一起的 LEGO 积木 |
| `torch.optim` | 更新权重的算法 | 机器学习里「学习」的那部分 |
| `autograd` | 自动计算梯度 | 自动帮你做微积分 |
| `DataLoader` | 高效喂数据 | 传送训练数据的流水线 |

## 安装 — 逐步进行

```bash
# 第 1 步：创建虚拟环境
python -m venv gpt_env

# 第 2 步：激活
source gpt_env/bin/activate          # Mac/Linux
# gpt_env\Scripts\activate           # Windows

# 第 3 步：安装 PyTorch（选对版本）
# 仅 CPU（默认，到处可用）：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon（M1/M2/M3）：
# pip install torch torchvision torchaudio

# NVIDIA GPU（CUDA 11.8）：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# NVIDIA GPU（CUDA 12.1 — RTX 40 系列等新卡）：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 第 4 步：安装其余包
pip install tiktoken datasets numpy matplotlib

# 第 5 步：验证一切正常
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 各库的作用（详解）

| 库 | 作用 | 为什么需要 |
|---|---|---|
| **torch** | PyTorch 核心：张量、GPU 运算、autograd | 基础——其他一切都建立在此之上 |
| **tiktoken** | OpenAI 的快速 BPE 分词器 | 与 GPT-3.5/4 相同的分词器。Rust 编写，极快 |
| **datasets**（HuggingFace） | 下载并缓存训练数据 | 省得我们手动下载和解析 Wikipedia |
| **numpy** | CPU 上的快速数值数组 | 用于快速数据处理（大部分仍由 PyTorch 承担） |
| **matplotlib** | 绘制图表 | 可视化训练 loss——模型在学习吗？ |
| **math**（内置） | sqrt、sin、cos、pi | 位置编码所需的数学常数 |
| **time**（内置） | 测量耗时 | 跟踪训练速度（tokens/秒） |
| **os**（内置） | 创建目录、保存文件 | 保存模型 checkpoint，避免丢失进度 |

## 完整的 import 代码块

```python
# ===== 是什么：标准 Python 库 =====
import math              # 为什么：sqrt()、sin()、cos() 用于位置编码数学
import time              # 为什么：测量训练速度（每秒 token 数）
import os                # 为什么：创建目录，保存/加载模型 checkpoint 文件
from dataclasses import dataclass  # 为什么：简洁的配置类——不用乱糟糟的字典

# ===== 是什么：NumPy — CPU 数组库 =====
import numpy as np       # 为什么：CPU 数组上的快速数值运算
                         #      （多用于快速数据检查，而非重活）

# ===== 是什么：PyTorch — 神经网络框架 =====
import torch             # 为什么：核心库——张量、GPU 支持、autograd
import torch.nn as nn               # 为什么：神经网络构建块：
                                     #      Linear（全连接层）、Embedding（查表）、
                                     #      Dropout（正则化）、ModuleList（堆叠层）
import torch.nn.functional as F     # 为什么：forward() 中使用的无状态函数：
                                     #      softmax（转为概率）、
                                     #      cross_entropy（衡量预测误差）、
                                     #      silu（SwiGLU 激活函数）
from torch.utils.data import Dataset, DataLoader  # 为什么：高效数据流水线
#                                  Dataset = 定义如何加载一个样本
#                                  DataLoader = 组 batch、打乱、预取

# ===== 是什么：tiktoken — OpenAI 的快速 BPE 分词器 =====
import tiktoken          # 为什么：与 GPT-3.5/GPT-4 相同的字节对编码（BPE）分词器
                         #      Rust 编写，比纯 Python 分词器快约 100 倍
                         #      高效处理 5 万+ 词表

# ===== 是什么：HuggingFace datasets — 下载训练文本 =====
from datasets import load_dataset    # 为什么：一行代码下载 WikiText-103
                                     #      处理缓存（只下载一次）、
                                     #      流式（数据集太大放不进磁盘时）、
                                     #      以及格式转换

# ===== 是什么：matplotlib — 绘制 loss 曲线 =====
import matplotlib.pyplot as plt      # 为什么：可视化训练进度
                                     #      loss 在下降吗？是否平台期？
                                     #      一图胜千行日志

# ===== 是什么：快速验证 =====
# 为什么：写 500 行代码之前先测试环境。
#      现在发现缺 import，能省几小时调试。
print("所有 import 就绪！")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**预期输出（有 GPU）：**
```
所有 import 就绪！
PyTorch 版本: 2.1.0
CUDA 可用:  True
GPU:             NVIDIA GeForce RTX 3090
GPU 显存:      24.0 GB
```

**预期输出（仅 CPU）：**
```
所有 import 就绪！
PyTorch 版本: 2.1.0
CUDA 可用:  False
```

若看到 GPU 输出，就可以开始训练。若只有 CPU，训练也能跑——只是更慢。无论哪种情况，都可以继续。

---

## 如何理解本指南的其余部分

每一章都遵循这个模式：

1. **类比** — 用通俗语言解释概念（像教 5 岁小孩一样）
2. **数学** — 展示实际公式以及它们为何有效
3. **代码** — 每一行都标注「是什么」和「为什么」
4. **可视化** — 用图或算例展示数据如何流动

若感到迷失，回到类比部分。若代码令人应接不暇，先看「是什么/为什么」注释——它们设计成可以自上而下像故事一样读。

---

**上一章：** [第 0 章 — 概览](00_overview.md)
**下一章：** [第 2 章 — 分词](02_tokenization.md)
