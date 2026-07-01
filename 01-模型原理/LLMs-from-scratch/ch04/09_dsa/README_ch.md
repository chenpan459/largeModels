# DeepSeek 稀疏注意力（DSA）

本附加材料实现了 [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 中引入、并首次发布于实验性版本 [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) 中的 DeepSeek 稀疏注意力（DSA）机制。

下面的概述遵循了 [From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates](https://magazine.sebastianraschka.com/p/technical-deepseek) 一文中对 DSA 的讨论。

&nbsp;
## 简介

标准的因果自注意力会针对每个查询关注所有先前的 token，随着序列长度 L 的增加，计算量呈 O(L²) 增长，KV 缓存呈 O(L) 增长。

[滑动窗口注意力（SWA）](../06_swa) 已经表明，将注意力限制在一个固定的局部窗口内可以大幅降低这一成本。在 SWA 中，每个查询 token 只关注附近先前 token 的一段局部范围。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/09.png" alt="Sliding window attention" width="800px" />

*图 1：滑动窗口注意力将每个查询 token 限制在一个固定的局部上下文窗口内。*

&nbsp;

DSA 采用了与之相同的广义思路，即只关注先前 token 中的一个子集。不过，它用一种学习到的选择机制取代了固定窗口。对于每个查询 token，模型会为候选的过去 token 打分，并只保留最相关的那些。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/10.png" alt="DeepSeek Sparse Attention selected-token pattern" width="800px" />

*图 2：DeepSeek 稀疏注意力为每个查询 token 选择一个学习到的过去 token 子集。*

&nbsp;

### 架构概述

DSA 在标准注意力的基础上增加了两个组件。

**1. 闪电索引器（Lightning Indexer）**

对于每个查询 token $t$ 以及每个候选的过去 token $s$，索引器会计算一个标量相关性分数。此实现明确列出了参考代码中的缩放因子：

$$I_{t,s} = \sum_{j=1}^{H_I} \frac{w_{t,j}}{\sqrt{H_I}} \cdot \text{ReLU}\left(\frac{q_{t,j} \cdot k_s}{\sqrt{d_I}}\right)$$

其中：
- $H_I$ 是轻量级索引头的数量，
- $q_{t,j}$ 是 token $t$ 与头 $j$ 对应的索引器查询向量，
- $k_s$ 是过去 token $s$ 所共享的索引器键向量，
- $w_{t,j}$ 是一个学习得到的、按 $1 / \sqrt{H_I}$ 缩放的逐头门控值。

ReLU 会将负的点积贡献置零，而带门控的求和则将各个索引头的结果聚合为每个过去 token 的单一相关性分数。

在完整的 DeepSeek 模型中，索引器基于来自多头潜在注意力（MLA）的压缩 token 表示进行工作。为了保持简单，此文件夹中的 GPT 实现直接基于常规隐藏状态计算索引器的查询和键。

**2. Token 选择器（Token Selector）**

在计算出所有索引器分数之后，只保留得分最高的 top-K 个位置。所有其他位置会在标准 softmax*之前*被掩码为 −∞，因此模型实际上只关注 $k \ll L$ 个 token。

索引器中的 ReLU 并不是最终稀疏性的来源。由于这些分数是在多个索引头上累加得到的，大多数最终分数仍然可能是非零的。真正产生稀疏模式的是 token 选择器，因为它只保留 top-K 个位置。

在一个融合的生产环境实现中，这可以将注意力计算量从 O(L²) 降低到 O(L·k)。而这里的实现保留了标准的密集注意力分数矩阵，并在 softmax 之前应用 DSA 选出的 top-K 掩码。这使得选择逻辑便于检查，但无法带来融合内核所具有的计算节省。

下图总结了整个流程：闪电索引器为候选 token 打分，选择器保留 top-K 个位置，得到的掩码限制了常规的注意力 softmax。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/11.png" alt="DeepSeek Sparse Attention flowchart" width="700px" />

*图 3：DSA 首先对候选 token 打分，然后保留 top-K 个 token 用于最终的注意力掩码。*

&nbsp;
## 实现

`gpt_with_kv_dsa.py` 提供了：

| 类 | 说明 |
|---|---|
| `LightningIndexer` | 用于评估过去 token 相关性的轻量级多头打分器。 |
| `MultiHeadAttentionWithDSA` | 带有 DSA 稀疏掩码 + 可选 KV 缓存的标准 MHA。 |
| `GPTModel` | 使用 `MultiHeadAttentionWithDSA` 替换标准注意力的 GPT 风格模型。 |

此实现遵循了本仓库中其他附加材料的风格，可以作为独立脚本运行。其目的是让 DSA 机制在一个小型 GPT 风格模型中变得便于检查。它并未实现 DeepSeek 完整的 MLA 技术栈、融合稀疏内核，或特定于部署环境的优化。

&nbsp;
## 用法

```bash
uv run gpt_with_kv_dsa.py \
  --emb_dim 768 \
  --n_heads 12 \
  --n_layers 12 \
  --max_new_tokens 200 \
  --index_n_heads 4 \
  --index_head_dim 64 \
  --topk 64
```

关键参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--index_n_heads` | 4 | 轻量级索引头的数量（H_I）。 |
| `--index_head_dim` | 64 | 每个索引头的维度。 |
| `--topk` | 64 | 每个查询所关注的 token 数量（k）。对于较短的序列，会被限制在序列长度以内。 |

&nbsp;
## 与 DeepSeek V3.2 的关系

全尺寸的 DeepSeek-V3.2 模型将多头潜在注意力（MLA，参见 [../05_mla](../05_mla)）与 DSA 结合使用，并且索引器的查询是从共享的压缩潜在表示中推导出来的，而不是来自原始输入。DeepSeek-V3.2 与首次引入并测试 DSA 的 DeepSeek-V3.2-Exp 采用了相同的架构。

这里复现了其中的关键选择思路：一个廉价的、学习得到的点积打分器，会在注意力 softmax 之前，将每个查询限制在最相关的若干个 token 上。

下面给出的推理成本对比数据，为理解 DSA 在长上下文部署场景中的重要性提供了有用的背景信息。这些节省效果依赖于生产环境中的内核实现和服务基础设施，因此该图不应被解读为对本文件夹中教学性实现的基准测试。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/19.png" alt="Inference cost comparison for DeepSeek Sparse Attention" width="800px" />

*图 4：DeepSeek 报告的、DSA 在长上下文服务场景中带来的推理成本节省，图片来自 [DeepSeek V3.2 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)。*

&nbsp;
## 参考资料

- DeepSeek V3.2 技术报告：https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp 模型卡与参考代码：https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Sebastian Raschka 的文章 "From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates"：https://magazine.sebastianraschka.com/p/technical-deepseek
