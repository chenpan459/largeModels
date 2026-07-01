# 专家混合模型（MoE）

本附加材料展示了使用专家混合（MoE）层代替常规前馈（FFN）层时，（每个 token）所带来的内存节省。



&nbsp;
## 简介

MoE 的核心思想是用多个专家层替换 transformer 块中的每一个前馈模块，其中每个专家层本身也是一个前馈模块。这意味着我们用多个前馈块替换了单个前馈块，如下图所示。



&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="SWA" width="800px" />

transformer 块内部的前馈块（在上图中以深灰色块表示）通常占据了模型总参数中相当大的一部分。（请注意，transformer 块以及其中的前馈块，会在一个 LLM 中重复多次；以 DeepSeek-V3 为例，重复了 61 次。）

因此，用*多个*前馈块替换*单个*前馈块（如在 MoE 设置中所做的那样）会大幅增加模型的总参数量。然而，关键的技巧在于，我们并不会对每个 token 都使用（“激活”）全部专家。相反，路由器（router）会为每个 token 只挑选一小部分专家。

由于同一时刻只有少数专家处于活跃状态，MoE 模块通常被称为*稀疏*模块，这与始终使用完整参数集的*密集*模块形成对比。然而，通过 MoE 得到的庞大总参数量提升了 LLM 的容量，这意味着它在训练期间可以吸收更多知识。不过，这种稀疏性保证了推理的高效，因为我们不会同时使用全部参数。

例如，DeepSeek-V3 每个 MoE 模块拥有 256 个专家，总参数量为 6710 亿。然而在推理过程中，同一时刻只有 9 个专家处于活跃状态（1 个共享专家加上路由器选出的 8 个专家）。这意味着每一步 token 推理只使用了 370 亿参数，而不是全部 6710 亿。

DeepSeek-V3 的 MoE 设计中一个值得注意的特点是引入了共享专家。这是一个对每个 token 都始终保持活跃的专家。这个想法并不新鲜，早在 [2022 年的 DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 和 [2024 年的 DeepSeek MoE](https://arxiv.org/abs/2401.06066) 论文中就已经被提出。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE shared expert" width="500px" />

（这是一张来自 [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) 论文的带注释图示。）

&nbsp;

拥有共享专家的好处最早在 [DeepSpeed-MoE 论文](https://arxiv.org/abs/2201.05596)中被指出，该论文发现，与没有共享专家相比，这样做能提升整体建模性能。这很可能是因为常见或重复出现的模式不再需要由多个独立的专家分别去学习，从而让各专家有更多空间去学习更为专门化的模式。

&nbsp;
## 专家混合模型（MoE）内存节省

MoE 模型的内存节省主要来自于激活值存储和计算量的减少。在常规（密集）前馈层（FFN）中，每个 token 都会激活完整的中间维度。

相比之下，MoE 层每个 token 只会经过一小部分专家（例如，`num_experts` 个专家中的 `top_k` 个）。

使用 MoE 层时，每个 token 只有 `top_k` 个专家处于活跃状态，因此相对于总容量相同的密集 FFN，有效内存（以及计算量）大致按 `top_k / num_experts` 的比例进行缩放。


你可以使用此文件夹中的 [memory_estimator_moe.py](memory_estimator_moe.py) 脚本，针对不同的模型配置应用这个公式，看看使用 MoE 相比 FFN 能节省多少内存（请注意，这是针对单个 transformer 块计算的；要得到总的节省量，需要乘以模型中 transformer 块的数量）：

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense 
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

因此，根据上面的结果可以看出，如果我们有一个输入/输出维度（`emb_dim`）为 7,168、中间维度（`hidden_dim`）为 14,336 的 FFN，这一层大约拥有 3.08 亿个参数，并且在前向传播中这些参数全部处于活跃状态。

现在，如果我们使用一个总参数量大致相同（约 3.08 亿）的 MoE 层，其中共有 8 个专家、2 个专家处于活跃状态，那么每次前向传播中只有约 7700 万个参数处于活跃状态。

此外，在专家总数不变的情况下，专家数量越多，活跃参数的数量就越少，“节省”的比例也就越大：

&nbsp;

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="SWA" width="500px" />



&nbsp;

你可以通过以下命令重现该图：

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


&nbsp;
## MoE 代码示例

此文件夹中的 [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) 和 [gpt_with_kv_moe.py](gpt_with_kv_moe.py) 脚本提供了动手示例，用于在 GPT 模型实现的背景下比较常规 FFN 和 MoE 的内存使用情况。请注意，这两个脚本都使用了本页第一幅图中所展示的 [SwiGLU](https://arxiv.org/abs/2002.05202) 前馈模块（GPT-2 传统上使用的是 GELU）。

**注意：该模型未经训练，因此会生成无意义的文本。你可以在附加材料 [../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb) 中找到一个已训练好的 MoE 模型。**



首先，让我们用常规 FFN 运行一下模型：


```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

为了与 MoE 进行公平比较，我们必须缩小专家的规模。例如，如果我们使用 32 个专家，就需要将 `--hidden_dim` 设为 32768/32：


```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

可以看到，密集前馈层处理一个 token 大约需要 0.76 毫秒，并使用约 0.19 MB 的激活值（峰值接近 0.75 MB）；

而稀疏 MoE 层只保留了约 0.04 MB 的内存（峰值为 0.11 MB）。不过，这是以大约两倍的计算时间为代价的。（这里存在额外的路由开销，并且我的实现也未必是最高效的。）

在这两种情况下，整体生成过程的 GPU 内存使用峰值都约为 11.5 GB，因为两个版本加载的权重参数数量相同，KV 缓存大小也相同，而这两者才是主导内存占用的因素。

不管怎样，我们都可以从中看到这样一种权衡：MoE 将 FFN 的内存减少了大约 4-5 倍，同时使前馈计算时间大致翻倍。

请注意，如果我们一次处理更多的 token，例如使用大于 1 的批量大小（这里由于代码简化的原因我们没有使用批处理），那么节省效果会更加明显。
