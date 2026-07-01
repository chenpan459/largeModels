# 加速 LLM 训练的 PyTorch 性能技巧



请注意，本书以教育为目的编写，原始代码刻意保持简单。这是为了便于阅读并确保在不同硬件（包括 CPU 和 GPU）上的兼容性。不过，你可能对更多高级 PyTorch 和 GPU 特性感兴趣，以提升 LLM 训练性能。

本文件夹包含三个代码文件，演示第 5 章介绍的 LLM 和训练函数的性能优化：

1. [`00_orig.py`](00_orig.py)：第 5 章原始代码，用于 CPU 和单 GPU 训练。  
   ➤ 运行方式：`python 00_orig.py`

2. [`01_opt_single_gpu.py`](01_opt_single_gpu.py)：单 GPU 训练的优化版本。  
   ➤ 运行方式：`python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py)：使用分布式数据并行（DDP）的多 GPU 训练优化版本。  
   ➤ 运行方式：`torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`  
   （**注意：** 为保持与 `01_opt_single_gpu.py` 的改动最小，此脚本仅支持通过上述 `torchrun` 进行多进程。这意味着**不支持**通过 `python 02_opt_multi_gpu_ddp.py` 使用多 GPU。）

**这些修改将训练速度从 12,525 tokens/秒（单 A100）提升到 142,156 tokens/秒（单 A100）和 419,259 tokens/秒（4× A100）。**

我计划在未来撰写更详细的差异说明。目前，查看代码改进的最简单方式是在 Visual Studio Code 中打开文件，通过「Compare Selected」功能查看差异。

![VS compare](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch Tips](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)

&nbsp;
## 单 GPU 速度对比

如上所述，我计划在未来更详细地阐述这些改动。目前，本节以 tokens/秒为单位，简要概述每项修改的性能。所有实验均在 A100 GPU 上运行。

&nbsp;
### 基线

注意，`00_orig.py` 作为基线，除以下改动外没有显著修改，基本使用第 5 章代码：

- 4 倍更大的上下文长度（这解释了 `00_orig.py` 相比第 5 章相对较大的内存占用）；
- 4 倍 batch size 变化（也是 `00_orig.py` 相对较大内存占用的因素）；
- 更大的公版书籍以增加训练数据量。

超参数未针对最小化损失和减少过拟合进行优化，LLM 最后生成的文本可能不够 sophisticated；但这不应影响主要结论，即此处作为速度参考的 `tok/sec` 指标（越高越好）。

```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch version: 2.6.0+cu124
Using cuda
CUDA version: 12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
...
Allocated memory: 2.5069 GB
Reserved memory: 26.2617 GB
```

注意 `01_opt_single_gpu.py` 包含下面依次列出的所有修改。

对比始终基于上一节第一个 epoch 后的平均 tok/sec 和已分配内存。

&nbsp;
### 1. 动态创建因果掩码

- 不保存因果掩码，而是动态创建以减少内存使用（此处效果很小，但在 Llama 3.2 等支持 131k 输入 token 的长上下文模型中会累积）

Before:
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

After:
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 2. 使用 Tensor Core

- 使用 tensor core（仅适用于 A100 及更新的 Ampere GPU）

Before:
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

After:
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 3. 融合 AdamW 优化器

- 通过设置 `fused=True` 使用 `AdamW` 的融合内核

Before:
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

After:
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 4. DataLoader 中的 pinned memory

- 在 data loader 中使用 `pin_memory=True` 预分配并复用 GPU 内存

Before:
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

After:
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 5. 使用 bfloat16 精度

- 从 32 位 float 切换到 16 位 brain float（bfloat16）精度（更多内容请参阅我的[这篇文章](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights)）

Before:
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

After:
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

&nbsp;
### 6. 用 PyTorch 类替换从零实现代码

- 用 PyTorch 原生实现替换 LayerNorm 和 GeLU 的从零实现

Before:
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

After:
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

&nbsp;
### 7. 使用 FlashAttention

- 使用 PyTorch 带 FlashAttention 的自注意力函数，替代我们的从零实现多头注意力

Before:
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

After:
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

&nbsp;
### 8. 使用 `pytorch.compile`

- 使用 `torch.compile(model)`。注意，最初几次迭代总是较慢，之后才会加速。由于 `Avg tok/sec` 测量仅包含平均计算的第一行，我们现在使用 epoch 1 结束时的 `Step tok/sec`。

Before:
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

After:
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

<br>

---

**Windows 说明**

- 在 Windows 上编译可能比较棘手
- `torch.compile()` 使用 Inductor，它会 JIT 编译内核并需要可用的 C/C++ 工具链
- 对于 CUDA，Inductor 还依赖 Triton，可通过社区包 `triton-windows` 获取
  - 如果看到 `cl not found`，请[安装带「C++ 工作负载」的 Visual Studio Build Tools](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)，并从「x64 Native Tools」提示符运行 Python
  - 如果 CUDA 下看到 `triton not found`，请安装 `triton-windows`（例如 `uv pip install "triton-windows<3.4"`）。
- 对于 CPU，有读者建议遵循此 [PyTorch Inductor Windows 指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
  - 安装 Visual Studio 2022 时安装英文语言包以避免 UTF-8 错误很重要
  - 此外，代码需要通过「Visual Studio 2022 Developer Command Prompt」运行，而非 notebook
- 如果此设置比较棘手，可以跳过编译；**编译是可选的，所有代码示例不编译也能正常运行**

---

&nbsp;
### 9. 词汇表 padding

- 此处我们将词汇表大小从 50,257 略微增加到 50,304，即最接近的 64 的倍数。此技巧由我的前同事 Carlos Mocholi 建议，他提到这最初来自 Andrej Karpathy（可能来自[此帖](https://x.com/karpathy/status/1621578354024677377)）。Karpathy 的建议基于与 PyTorch 团队的交流，[Bertrand Maher](https://www.linkedin.com/feed/update/urn:li:activity:7309569006057795584?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A7309569006057795584%2C7309754284185669632%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287309754284185669632%2Curn%3Ali%3Aactivity%3A7309569006057795584%29) 提到了关于 `torch.compile` 的建议。关于此主题的好资源是 [NVIDIA 的 tensor shape 指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape)，其中 batch size 和线性层维度通常选择为特定值的倍数。此外，vocab-padding 技巧在 NVIDIA Megatron 团队的 2019 年 [Megatron-LM 论文](https://arxiv.org/abs/1909.08053) 中已有描述。

Before:
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

After:
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

&nbsp;
### 10. 增大 batch size

- 最后，我们将 batch size 增大到 GPU 支持的最大 2 的幂

Before:
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

After:
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

&nbsp;
## 多 GPU 速度对比

这可能不是完全公平的对比，因为我们现在使用 4 个 GPU 而非 1 个，但使用分布式数据并行——当训练不受 GPU 内存限制时最快的多 GPU 技术——当然可以带来显著加速：

Before（单 GPU）：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

After（4 GPU）：
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`
