# 分组查询注意力（GQA）

本附加材料展示了与常规多头注意力（MHA）相比，使用分组查询注意力（GQA）所带来的内存节省。

&nbsp;
## 简介

近年来，分组查询注意力（GQA）已成为多头注意力（MHA）的新标准替代方案，是一种计算和参数效率更高的选择。请注意，这并不是一个新概念，它可以追溯到 2023 年的论文 [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)。甚至连经典的 Llama 2 系列中较大的变体也使用了它。

下面是对 GQA 的简要总结。与 MHA 中每个头都有一套自己的键和值不同，为了减少内存使用，GQA 将多个头分组，使它们共享同一套键和值投影。

例如，如下图进一步说明的那样，如果有 3 个键值组和 6 个注意力头，那么头 1 和头 2 共享一组键和值，而头 3 和头 4、以及头 5 和头 6 则分别共享另外两组。

&nbsp;

![GQA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)

&nbsp;

这种键和值的共享减少了键和值计算的总次数，从而降低了内存使用量并提升了效率。

因此，总结来说，GQA 的核心思想就是通过在多个查询头之间共享键和值头，来减少键和值头的数量。这样做（1）降低了模型的参数量，并且（2）在推理过程中减少了键和值张量所占用的内存带宽，因为需要从 KV 缓存中存取的键和值数量更少了。

虽然 GQA 主要是针对 MHA 的一种计算效率折衷方案，但消融研究（例如[原始 GQA 论文](https://arxiv.org/abs/2305.13245)以及 [Llama 2 论文](https://arxiv.org/abs/2307.09288)中的研究）表明，就 LLM 建模性能而言，它的表现与标准 MHA 相当。

不过，这是建立在键值组数量经过仔细选择的前提下的。在极端情况下，如果所有注意力头共享同一个键值组（即所谓的多查询注意力），内存使用量会进一步大幅下降，但建模性能可能会受到影响。（而在另一个极端情况下，如果我们将键值组的数量设置为等于查询头的数量，那我们就回到了标准的多头注意力。）

&nbsp;
## GQA 内存节省

内存节省主要体现在 KV 存储上。我们可以用以下公式计算 KV 存储的大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

你可以使用此文件夹中的 [memory_estimator_gqa.py](memory_estimator_gqa.py) 脚本，针对不同的模型配置应用这个公式，看看使用 GQA 相比 MHA 能节省多少内存：

```bash
➜ uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== Config ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 128
GQA n_kv_heads   : 8

==== KV-cache totals across all layers ====
MHA total KV cache  : 17.18 GB
GQA total KV cache  : 4.29 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```

下图进一步展示了在不同键值组大小下，使用 GQA 相比 MHA 随上下文长度变化所节省的内存：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="GQA" width="500px" />

&nbsp;

你可以通过 `uv run plot_memory_estimates_gqa.py` 重现该图。

&nbsp;
## GQA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) 脚本提供了动手示例，用于在 GPT 模型实现的背景下比较 MHA 和 GQA 的内存使用情况。

请注意，GQA 也被用于 [Llama 3](../../ch05/07_gpt_to_llama)、[Gemma 3](../../ch05/12_gemma3) 和 [Qwen3](../../ch05/11_qwen3) 附加材料中。不过，为了简单起见，此文件夹中的代码脚本修改的是传统上并未使用 GQA 的 GPT 架构。

请注意，该模型未经训练，因此会生成无意义的文本。不过，你可以将其作为第 5-7 章中标准 GPT 模型的直接替代品，并对其进行训练。

此外，此实现使用了[另一个附加章节](../03_kv-cache)中介绍的 KV 缓存，因此内存节省效果更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

Time: 516.33 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

之所以我们没有看到像上面图中那样大的节省，原因有二：

1. 我使用了较小的配置，以便模型能在合理的时间内完成生成。
2. 更重要的是，我们这里考察的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（不过这是另一个需要单独分析的话题）。
