# 滑动窗口注意力（SWA）

本附加材料展示了与常规多头注意力（MHA）相比，使用滑动窗口注意力（SWA）所带来的内存节省。



&nbsp;
## 简介

什么是滑动窗口注意力（SWA）？如果我们把常规的自注意力看作一种*全局*注意力机制（因为其中每个序列元素都可以访问其他所有序列元素），那么我们可以把 SWA 看作*局部*注意力，因为这里我们限制了当前查询位置周围的上下文大小。下图对此进行了说明。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="Sliding Window Attention" width="500px" />

如上图所示，每个 token 不再关注所有先前的 token，而只关注其位置周围一个固定大小的局部窗口。这种局部化的注意力大幅降低了 KV 缓存的规模。

在本简介的其余部分，我们将结合 [Gemma 3](https://arxiv.org/abs/2503.19786)（在 [../../ch05/12_gemma3](../../ch05/12_gemma3) 中从零实现）来讨论 SWA。

滑动窗口注意力最早是在 [2020 年的 LongFormer 论文](https://arxiv.org/abs/2004.05150)中提出的，但我们之所以关注 Google 的 Gemma 系列模型，是因为它们是非常优秀的开放权重模型，证明了滑动窗口注意力在近期强大的模型中确实是一种可行的方案。

[Gemma 2](https://arxiv.org/abs/2408.00118) 采用了一种混合方式，将局部（滑动窗口）注意力层和全局注意力层以 1:1 的比例结合起来。每个 token 可以关注一个 4k token 的上下文窗口。之所以采用这种 1:1 的混合方式，是因为它在效率和全局上下文建模之间取得了平衡，因为一个仅使用局部注意力的 LLM 可能会限制过度。

而 [Gemma 3](https://arxiv.org/abs/2503.19786) 则在这一设计基础上进一步追求效率。它在滑动窗口层和全注意力层之间采用了 5:1 的比例，也就是说，每 5 个局部注意力层就有 1 个全局层。此外，滑动窗口大小也从 Gemma 2 中的 4096 个 token 减小到了 Gemma 3 中的 1024 个 token。

有趣的是，Gemma 3 技术报告中的消融研究表明，这些改动对整体模型质量的影响很小。换句话说，通过滑动窗口注意力所实现的可观内存和计算节省，只带来了极小的建模性能损失。



&nbsp;
## 滑动窗口注意力（SWA）内存节省

内存节省主要体现在 KV 存储上。我们可以用以下公式计算 KV 存储的大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

使用 SWA 时，我们将上面公式中的序列长度（seqlen）替换为窗口大小 W。因此，使用滑动窗口注意力时，我们将 KV 缓存大小按 "W / seqlen" 的比例缩减了。（请注意，为了简单起见，这里假设每一层都使用了滑动窗口注意力。）


你可以使用此文件夹中的 [memory_estimator_swa.py](memory_estimator_swa.py) 脚本，针对不同的模型配置应用这个公式，看看使用 SWA 相比 MHA 能节省多少内存：

```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
GQA + SWA (Ratio: 5:1) : 0.78 GB
```

请注意，Gemma 3 将 SWA 与 GQA 结合使用。

下图进一步展示了在不同上下文长度下，使用 SWA 相比 MHA 所节省的内存：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="800px" />

&nbsp;

你可以通过以下命令重现该图：

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


&nbsp;
## SWA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_swa.py](gpt_with_kv_swa.py) 脚本提供了动手示例，用于在 GPT 模型实现的背景下比较 MHA 和 SWA 的内存使用情况。

请注意，如前所述，SWA 也可以与 MLA 和 GQA 结合使用，但为了简单起见，这里没有这样做。

请注意，该模型未经训练，因此会生成无意义的文本。不过，你可以将其作为第 5-7 章中标准 GPT 模型的直接替代品，并对其进行训练。

此外，此实现使用了[另一个附加章节](../03_kv-cache)中介绍的 KV 缓存，因此内存节省效果更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # like Gemma 3

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

之所以我们没有看到像上面图中那样大的节省，原因有二：

1. 我使用了较小的配置，以便模型能在合理的时间内完成生成。
2. 更重要的是，我们这里考察的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（不过这是另一个需要单独分析的话题）。
