# 多头潜在注意力（MLA）

本附加材料展示了与常规多头注意力（MHA）相比，使用多头潜在注意力（MLA）所带来的内存节省。

&nbsp;
## 简介

在 [../04_gqa](../04_gqa) 中，我们讨论了分组查询注意力（GQA）作为针对 MHA 的一种计算效率折衷方案。消融研究（例如[原始 GQA 论文](https://arxiv.org/abs/2305.13245)以及 [Llama 2 论文](https://arxiv.org/abs/2307.09288)中的研究）表明，就 LLM 建模性能而言，它的表现与标准 MHA 相当。

现在，被用于 [DeepSeek V2、V3 和 R1](https://arxiv.org/abs/2412.19437) 的多头潜在注意力（MLA）提供了一种不同的内存节省策略，它同样与 KV 缓存搭配得特别好。与 GQA 共享键和值头的方式不同，MLA 会先将键和值张量压缩到一个更低维度的空间中，然后再将其存入 KV 缓存。

在推理时，如下图所示，这些被压缩的张量会在使用前被投影回其原始大小。这增加了一次额外的矩阵乘法，但降低了内存使用量。

&nbsp;

![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

&nbsp;

（顺带一提，查询向量也会被压缩，但仅在训练期间进行，推理时不会。）

顺便说一下，正如前面提到的，MLA 在 DeepSeek V3 中并不是什么新事物，因为它的前身 [DeepSeek V2](https://arxiv.org/abs/2405.04434) 就已经使用了（甚至是引入了）它。此外，V2 论文中还包含了一些有趣的消融研究，或许可以解释为什么 DeepSeek 团队选择了 MLA 而不是 GQA（见下图）。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" />

&nbsp;

如上图所示，GQA 的表现似乎不如 MHA，而 MLA 的建模性能则优于 MHA，这很可能就是 DeepSeek 团队选择 MLA 而非 GQA 的原因。（如果能看到 MLA 和 GQA 在“每个 token 的 KV 缓存”节省方面的对比，那会很有意思！）

在进入下一个架构组件之前，让我们对这一节做个总结：MLA 是一个巧妙的技巧，它在减少 KV 缓存内存使用的同时，建模性能甚至还略微优于 MHA。

&nbsp;
## MLA 内存节省

内存节省主要体现在 KV 存储上。我们可以用以下公式计算 KV 存储的大小：

bytes ≈ batch_size × seqlen × n_layers × latent_dim × bytes_per_elem

相比之下，MHA 的 KV 缓存内存计算方式如下：

bytes ≈ batch_size × seqlen × n_layers × embed_dim × 2 (K,V) × bytes_per_elem

这意味着，在 MLA 中，我们把 "embed_dim × 2 (K,V)" 缩减为了 "latent_dim"，因为如前面图中所示，我们只存储压缩后的潜在表示，而不是完整的键和值向量。



你可以使用此文件夹中的 [memory_estimator_mla.py](memory_estimator_mla.py) 脚本，针对不同的模型配置应用这个公式，看看使用 MLA 相比 MHA 能节省多少内存：

```bash
➜ uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

请注意，上面的压缩比例（`--emb_dim 2048 -> latent_dim 1024`）是为了达到与 GQA 类似的节省效果。在实践中，压缩比例是一个需要仔细研究的超参数，因为将 `latent_dim` 设得太小会对建模性能产生负面影响（这与 GQA 中把 `n_kv_groups` 设得过大是类似的道理）。

下图进一步展示了在不同 `latent_dim` 取值下，使用 MLA 相比 MHA 随上下文长度变化所节省的内存：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" />

&nbsp;

你可以通过 `uv run plot_memory_estimates_mla.py` 重现该图。



&nbsp;
## MLA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_mla.py](gpt_with_kv_mla.py) 脚本提供了动手示例，用于在 GPT 模型实现的背景下比较 MHA 和 MLA 的内存使用情况。

这里的 MLA 代码灵感来自 [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla) 的实现。

请注意，MLA 也可以与 [GQA](../04_gqa) 结合使用，但为了简单起见，我在这里没有这样做。（就目前而言，我也还不知道有哪个知名的 LLM 是这样做的。）

同样请注意，该模型未经训练，因此会生成无意义的文本。不过，你可以将其作为第 5-7 章中标准 GPT 模型的直接替代品，并对其进行训练。

最后，此实现使用了[另一个附加章节](../03_kv-cache)中介绍的 KV 缓存，因此内存节省效果更加明显。

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
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× compression

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

之所以我们没有看到像上面图中那样大的节省，原因有二：

1. 我使用了较小的配置，以便模型能在合理的时间内完成生成。
2. 更重要的是，我们这里考察的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（不过这是另一个需要单独分析的话题）。
