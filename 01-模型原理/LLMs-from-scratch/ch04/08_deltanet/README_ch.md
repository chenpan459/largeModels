# 用于线性注意力的门控 DeltaNet（Gated DeltaNet）

最近，[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 和 [Kimi Linear](https://arxiv.org/abs/2510.26692) 提出了混合 transformer 架构，实现了相对于上下文长度呈线性扩展、而非二次方扩展的注意力机制替代方案。

Qwen3-Next 和 Kimi Linear 都采用了 3:1 的比例，也就是说，每三个使用线性 Gated DeltaNet 变体的 transformer 块，就有一个块使用全注意力，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next versus Kimi Linear">



&nbsp;

## 简介与概述

Gated DeltaNet 是一种线性注意力变体，其灵感来自循环神经网络，其中包含了源自 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文的门控机制。从某种意义上说，Gated DeltaNet 就是带有 Mamba 风格门控的 DeltaNet，而 DeltaNet 本身则是一种线性注意力机制。

Kimi Linear 通过 Kimi Delta Attention（KDA）机制修改了 Qwen3-Next 的线性注意力机制，KDA 本质上是对 Gated DeltaNet 的一种改进。Qwen3-Next 使用的是标量门控（每个注意力头一个值）来控制记忆的衰减速率，而 Kimi Linear 则将其替换为针对每个特征维度的逐通道门控。据作者所说，这种方式对记忆提供了更精细的控制，进而提升了长上下文推理能力。

此外，对于全注意力层，Kimi Linear 用多头潜在注意力（MLA）取代了 Qwen3-Next 的门控注意力层（本质上是带有输出门控的标准多头注意力层）。这与我们之前在 DeepSeek V3/R1 部分讨论过的 MLA 机制相同，只是多了一个额外的门控。（回顾一下，MLA 会压缩键/值空间，以减小 KV 缓存的大小。）

Kimi Linear 中的 MLA 并未使用该门控，这是有意为之的设计，目的是让作者可以将该架构与标准 MLA 进行更直接的比较，不过他们[表示](https://x.com/yzhang_cs/status/1984631714464088563)计划在未来添加该门控。

由于我们已经在 [../05_mla](../05_mla) 中实现了 MLA，本附加材料将重点放在 Gated DeltaNet 相关的内容上。


&nbsp;
## 门控注意力

在正式介绍 Gated DeltaNet 之前，让我们先简要谈谈门控。正如你在前面图中 Qwen3-Next 架构上半部分所看到的，Qwen3-Next 使用了“门控注意力”（gated attention）。这本质上就是常规的全注意力，再加上一个额外的 sigmoid 门控。

为便于说明，下面是我在第 3 章的 `MultiHeadAttention` 代码基础上，添加这种门控所做的简单修改：

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Add gate
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        ####################################################
        ### NEW: Add gate
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### NEW: Add gate
        context = context * torch.sigmoid(gate)
        ####################################################
        out = self.out_proj(context)
        return out
```



可以看到，在照常计算完注意力之后，模型会从同一输入中得到一个独立的门控信号，通过 sigmoid 函数将其限制在 0 到 1 之间，再将其与注意力输出相乘。这使得模型能够动态地放大或缩小某些特征。Qwen3-Next 的开发者[表示](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)，这有助于提升训练稳定性：

> [...] the attention output gating mechanism helps eliminate issues like Attention Sink and Massive Activation, ensuring numerical stability across the model.
>
> （译文：……注意力输出门控机制有助于消除诸如 Attention Sink（注意力汇聚）和 Massive Activation（大规模激活）之类的问题，确保整个模型的数值稳定性。）


&nbsp;
## Gated DeltaNet

那么，Gated DeltaNet 究竟是什么？Gated DeltaNet（门控 Delta 网络）是 Qwen3-Next 中的线性注意力层，旨在作为标准 softmax 注意力的一种替代方案。如前所述，它改编自 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文。

Gated DeltaNet 最初是作为 Mamba2 的改进版本被提出的，它将 Mamba2 的门控衰减机制与 delta 规则结合了起来。

Mamba 是一种状态空间模型（作为 transformer 的一种替代方案），这是一个值得未来单独讨论的大话题。

delta 规则部分指的是，计算新值与预测值之间的差值（delta，Δ），用它来更新一个被当作记忆状态使用的隐藏状态（稍后会详细介绍）。

（补充说明：熟悉经典机器学习文献的读者，可以将其类比为受生物学启发的 Hebbian 学习："一起激活的神经元会连接在一起。" 它本质上可以看作是感知机更新规则以及基于梯度下降的学习方式的一个前身，只是没有监督信号。）

Gated DeltaNet 拥有一个与前面讨论的门控注意力中类似的门，不同之处在于它使用的是 SiLU 而非逻辑 sigmoid 激活函数，如下图所示。（选择 SiLU 很可能是为了在梯度流动和稳定性方面优于标准 sigmoid。）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="Gated DeltaNet" width=500px>

不过，如上图所示，Gated DeltaNet 中的“门控”（Gated）一词还指代了另外几个门：

- `α`（衰减门）控制记忆随时间衰减或重置的速度，
- `β`（更新门）控制新输入对状态的修改强度。

在代码中，上图所描绘的 Gated DeltaNet 的一个简化版本（不含卷积混合）可以实现如下（该代码的灵感来自 Qwen3 团队的[官方实现](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835)）。

（请注意，一些实现将衰减门称为 `gk`（步骤 k 的门控），其中 `exp(gk)` 对应于论文中的 $\alpha_t$。为了明确这种对应关系，下面的代码片段将对数空间中的门控 `alpha_log` 与经过指数化处理的衰减 `alpha` 区分开来。）


```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Gates for delta rule and output gating
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # Note: The decay gate alpha corresponds to
        # A_log + W_alpha(x) + dt_bias
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        # We could implement this as
        # W_alpha = nn.Linear(d_in, num_heads, bias=True)
        # but the bias is separate for interpretability and
        # to mimic the official implementation

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        ####################################################
        ### NEW: Compute delta rule gates
        beta = torch.sigmoid(self.W_beta(x))
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        alpha = alpha_log.exp()
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # NEW

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)

        ####################################################
        ### NEW: QKNorm-like normalization for delta rule
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)
        ####################################################

        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        ####################################################
        ### NEW: Gated delta rule update
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
            ####################################################
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        ### NEW: Apply RMSNorm and SiLU gate
        context = self.norm(context)
        context = context * F.silu(gate)
        ####################################################

        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

（请注意，为简单起见，我省略了 Qwen3-Next 和 Kimi Linear 中用来使代码更易读、并聚焦于循环特性的卷积混合部分。）

因此，正如我们所看到的，上面的实现与标准（或门控）注意力之间存在许多不同之处。

在门控注意力中，模型会在所有 token 之间计算常规的注意力（每个 token 都会关注或查看其他所有 token）。然后，在得到注意力输出后，一个门（sigmoid）决定保留该输出的多少部分。需要注意的是，这本质上仍然是相对于上下文长度呈二次方扩展的常规缩放点积注意力。

回顾一下，缩放点积注意力的计算方式为 softmax(QKᵀ)V，其中 Q 和 K 是 *n* 行 *d* 列的矩阵，*n* 是输入 token 的数量，*d* 是嵌入维度。因此，QKᵀ 会得到一个 *n* 行 *n* 列的注意力矩阵，再与一个 *n* 行 *d* 列的值矩阵 V 相乘：

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="Quadratic attention" width=500px />

在 Gated DeltaNet 中，不存在 *n* 行 *n* 列的注意力矩阵。相反，模型会逐个处理 token。它维护一个运行中的记忆（一个状态），每当有新 token 到来时，该状态就会被更新。这正是下面代码所实现的内容，其中 `S` 就是在每个时间步 *t* 被递归更新的状态。

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

而这些门控则控制着这个记忆的变化方式：

- α (`alpha`) 调节遗忘（衰减）多少旧记忆。

- β (`beta`) 调节当前时间步 *t* 的 token 对记忆的更新程度。

（而最终的输出门——上面的代码片段中未展示——与门控注意力中的门类似；它控制着保留多少输出。）

因此，从某种意义上说，Gated DeltaNet 中的这种状态更新方式，与循环神经网络（RNN）的工作方式相似。其优势在于，它相对于上下文长度呈线性扩展（通过 for 循环实现），而非二次方扩展。

这种循环状态更新方式的缺点在于，与常规（或门控）注意力相比，它牺牲了源自完整成对注意力所带来的全局上下文建模能力。

Gated DeltaNet 在一定程度上仍然可以捕获上下文信息，但必须经过记忆（*S*）这一瓶颈。这个记忆的大小是固定的，因此更加高效，但它会像 RNN 一样，把过去的上下文压缩进单一的隐藏状态中。

这就是为什么 Qwen3-Next 和 Kimi Linear 架构没有把所有注意力层都替换为 DeltaNet 层，而是采用了前面提到的 3:1 比例。

&nbsp;
## DeltaNet 内存节省

在上一节中，我们讨论了相对于上下文长度而言，DeltaNet 相比全注意力在计算复杂度方面所具有的线性（而非二次方）扩展优势。

除了线性计算复杂度之外，DeltaNet 的另一大优势是内存节省，因为 DeltaNet 模块不会使 KV 缓存增大。（有关 KV 缓存的更多信息，请参见 [../03_kv-cache](../03_kv-cache)。）相反，如前所述，它们维护的是一个固定大小的循环状态，因此内存不会随着上下文长度的增长而增长。

对于常规的多头注意力（MHA）层，我们可以按如下方式计算 KV 缓存大小：

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

（这里的乘数 2 是因为我们在缓存中同时存储了键和值。）

对于上面实现的简化版 DeltaNet，我们有：


```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

请注意，`KV_cache_DeltaNet` 的内存大小并不依赖于上下文长度（`n_tokens`）。此外，我们只存储了记忆状态 S，而不是分别存储的键和值，因此 `2 × bytes` 变成了 `bytes`。不过，请注意这里出现了一个二次方项 `d_head × d_head`。这来自于以下状态定义：

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

但这通常不需要太过担心，因为头维度通常相对较小。例如，在 Qwen3-Next 中它就是 128。

包含卷积混合的完整版本要更复杂一些，涉及卷积核大小等因素，但上面的公式应该已经足以说明 Gated DeltaNet 背后的主要趋势和设计动机。

我们可以通过以下辅助脚本，可视化不同上下文长度下的内存估算值及其节省情况：

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

请注意，上面的计算是将 `head_dim` 计算为 `emb_dim / n_heads`。也就是说，2048 / 16 = 128。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="Gated DeltaNet scaling" width=500px>
