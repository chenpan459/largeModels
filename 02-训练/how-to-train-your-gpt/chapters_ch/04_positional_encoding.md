# 第 4 章 — 位置编码：教会模型顺序

## 5 岁小孩也能懂的类比

看两句话：
- "The **dog** bit the **man**."  — 吓人
- "The **man** bit the **dog**."  — 奇怪

同样的词，顺序不同 -> **意思完全不同**。

但 Transformer **同时**读所有词（不像人类一个一个读）。它**不知道**哪个词在前！因此必须在喂给模型之前**给每个词盖上位置标记**。

## 位置编码的三代演进

| 方法 | 原理 | 优点 | 缺点 | 使用者 |
|---|---|---|---|---|
| **Learned（可学习）** | 每个位置有独立可学习向量 | 简单、灵活 | 无法处理比训练更长的序列 | GPT-2、BERT |
| **Sinusoidal（正弦）** | 按位置的固定 sin/cos 波 | 任意长度可用 | 相对位置较弱 | 原始 Transformer |
| **RoPE（旋转位置嵌入）** | 按位置角度旋转 Q、K 向量 | 相对位置完美、任意长度 | 略复杂 | LLaMA、Mistral、Qwen、Gemma |
| **ALiBi** | 按距离给 attention 分数加偏置 | 无可学习参数、极快 | 表达力较弱 | BLOOM、MPT |

## 现代方案：旋转位置嵌入（RoPE）

RoPE 不是把位置数**加**到嵌入上，而是按位置相关的角度**旋转** query 和 key 向量。

### 数学直觉

在 2D 中，把向量 `(x, y)` 旋转角度 `θ` 得到：
```
x' = x*cos(θ) - y*sin(θ)
y' = x*sin(θ) + y*cos(θ)
```

RoPE 对 query 和 key 向量的**每一对维度**都做此操作。位置 `p`、维度对 `2i, 2i+1` 的旋转角为：

```
θ(p, i) = p / (10000^(2i/d_model))
```

**关键洞察：** 角度取决于 `p`（位置）和 `i`（维度对索引）。低维度对旋转**快**（捕捉局部词关系）。高维度对旋转**慢**（捕捉长程关系）。

### 数值算例

用 tiny 模型跟踪 RoPE：`d_model=4`，处理位置 `p=1`：

**第 1 步：计算每个维度对的频率**

```
维度对 0（维 0,1）：freq = 1 / 10000^(0/4)   = 1 / 1       = 1.000
维度对 1（维 2,3）：freq = 1 / 10000^(2/4)   = 1 / 10000^0.5 = 1 / 100 = 0.010
```

**第 2 步：计算位置 p=1 的旋转角**

```
维度对 0 角度：θ₀ = p * freq₀ = 1 * 1.000 = 1.000 弧度（≈ 57.3°）
维度对 1 角度：θ₁ = p * freq₁ = 1 * 0.010 = 0.010 弧度（≈ 0.57°）
```

**第 3 步：对位置 1 的 query 向量应用旋转**

```
RoPE 前：q₁ = [0.8, 0.3, -0.5, 0.2]

旋转维度对 0（维 0,1），角度 57.3°：
  dim0' = 0.8*cos(1.0) - 0.3*sin(1.0) = 0.8*0.540 - 0.3*0.842 = 0.432 - 0.253 = 0.179
  dim1' = 0.8*sin(1.0) + 0.3*cos(1.0) = 0.8*0.842 + 0.3*0.540 = 0.674 + 0.162 = 0.836

旋转维度对 1（维 2,3），角度 0.57°：
  dim2' = -0.5*cos(0.01) - 0.2*sin(0.01) = -0.5*1.000 - 0.2*0.010 = -0.500 - 0.002 = -0.502
  dim3' = -0.5*sin(0.01) + 0.2*cos(0.01) = -0.5*0.010 + 0.2*1.000 = -0.005 + 0.200 = 0.195

RoPE 后：q₁' = [0.179, 0.836, -0.502, 0.195]
```

再看位置 1 和 3 会发生什么：

```
位置 1：θ₀ = 1.0 rad，θ₁ = 0.01 rad
位置 3：θ₀ = 3.0 rad，θ₁ = 0.03 rad

点积 q₁ · k₃ 取决于**差值**：
  Δθ₀ = 3.0 - 1.0 = 2.0 rad
  Δθ₁ = 0.03 - 0.01 = 0.02 rad
  
该差值**只**取决于 (3-1)=2，即相对距离！
绝对位置不重要——只有相距多远才重要。
```

这就是 RoPE 的精妙之处：位置 `i` 与 `j` 之间的 attention 分数**只**取决于相对距离 `(j-i)`，而非绝对位置。

### 为什么 theta=10000？

基频 `theta = 10000` 控制频率的「分布」：

```
低 theta（如 100）：
  - 各维度对旋转方式相似
  - 模型更「位置无关」——更适合长上下文
  - 但损失细粒度位置分辨率

高 theta（如 100000）：
  - 各维度旋转速度差异很大
  - 更好区分相邻位置
  - 但在极长上下文上吃力

10000 是经验上平衡这些权衡的值。
```

### 超出训练长度的上下文扩展

若在 2048 token 上训练，推理时想用 4096 怎么办？

**问题：** RoPE 为位置 0–2047 预计算。位置 3000 从未见过。

**方案：**
| 方法 | 原理 | 质量 |
|---|---|---|
| **Linear interpolation（线性插值）** | Position / scale（如 2 倍长度用 p/2） | 尚可，损失分辨率 |
| **NTK-aware scaling** | 按频率不同缩放 theta | 较好 |
| **YaRN** | NTK + 温度缩放 | 最好（生产在用） |
| **Retrain（重训）** | 直接在更长序列上训练 | 完美但昂贵 |

对我们小规模训练不重要——但要知道这是生产模型的热门研究方向。

## RoPE 代码 — 逐行注释

```python
import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    是什么：旋转位置嵌入（RoPE）。
    为什么：不是把位置信息**加**到嵌入上，
         而是按位置相关角度**旋转** Q 和 K 向量。
         点积 q_i · k_j 则**只**取决于 (j-i)，
         这正是 attention 应该关心的。

         论文："RoFormer"（Su et al., 2021）
         用于：LLaMA 1/2/3、Mistral、Mixtral、Qwen 1/2、Gemma

         概览：
         1. 对每对维度 (0,1)、(2,3)、(4,5)、...
         2. 按 angle = position * frequency 旋转
         3. 低维旋转快（局部位置）
            高维旋转慢（全局位置）
         4. 点积自然依赖相对距离
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        是什么：预计算旋转频率以便快速查表。

        参数：
            d_model:     头维度（如 GPT-2 为 64）。必须为偶数。
            max_seq_len: 为位置 0..max_seq_len-1 预计算角度。
            theta:       基频。10000 是标准值。控制快、慢旋转频率的分布。
        """
        super().__init__()

        # 是什么：验证 d_model 为偶数（必须有成对维度可旋转）
        assert d_model % 2 == 0, (
            f"d_model ({d_model}) 必须为偶数才能使用 RoPE。"
            f"每一对维度都需要一个配对维度来一起旋转。"
        )

        # 是什么：创建维度索引：[0, 2, 4, ..., d_model-2]
        # 为什么：每对 (2i, 2i+1) 共享同一旋转频率。
        #      只需一半索引，因为成对共享。
        dim_indices = torch.arange(0, d_model, 2).float()

        # 是什么：计算旋转频率
        # 为什么：theta_i = 1 / (theta ^ (2i / d_model))
        #
        #      i=0:  1 / 10000^(0/64)      = 1.0      → 快旋转（局部）
        #      i=30: 1 / 10000^(60/64)     ≈ 0.0001   → 慢旋转（全局）
        #
        #      这种多尺度方式意味着某些维度
        #      捕捉局部词序，另一些捕捉长程位置关系。
        inv_freq = 1.0 / (theta ** (dim_indices / d_model))

        # 是什么：为所有位置预计算角度
        # 为什么：训练时算 cos/sin 很贵。
        #      预计算并缓存一次快约 100 倍。
        positions = torch.arange(max_seq_len).float()     # [0, 1, 2, ..., 2047]

        # 是什么：外积：每个位置 × 每个频率
        #       freqs[p, i] = p * inv_freq[i] = 位置 p、维度对 i 的角度
        #       形状：[max_seq_len, d_model/2]
        freqs = torch.outer(positions, inv_freq)

        # 是什么：复制到完整维度
        # 为什么：每对 (2i, 2i+1) 同一角度，
        #      复制：[θ0, θ1, θ2, ...] -> [θ0, θ0, θ1, θ1, ...]
        emb = torch.cat([freqs, freqs], dim=-1)         # [max_seq_len, d_model]

        # 是什么：缓存所有位置的 cos 和 sin
        # 为什么：register_buffer 表示随 model.to(device) 移动，
        #      并保存在 model.state_dict() 中，但**不是**
        #      可训练参数（不需要梯度）。
        self.register_buffer("cos_cached", emb.cos())   # 各角度的 cos
        self.register_buffer("sin_cached", emb.sin())   # 各角度的 sin

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        是什么：为旋转公式准备向量。
        为什么：  旋转公式：x' = x*cos + rotate_half(x)*sin
        
              对向量 [x0, x1, x2, x3, x4, x5]：
              rotate_half 返回 [-x1, x0, -x3, x2, -x5, x4]
              
              为何有效：对 (x0, x1) 旋转角 θ：
                x0' = x0*cos(θ) - x1*sin(θ)   ← 对应：x0*cos + (-x1)*sin
                x1' = x0*sin(θ) + x1*cos(θ)   ← 对应：x1*cos + (x0)*sin
              
              因此 (x*cos + rotate_half(x)*sin) 同时对
              每对维度做旋转——无需循环！
        """
        x1 = x[..., : x.shape[-1] // 2]   # 前半：[x0, x2, x4, ...]
        x2 = x[..., x.shape[-1] // 2 :]   # 后半：[x1, x3, x5, ...]
        return torch.cat([-x2, x1], dim=-1)  # [-x1, x0, -x3, x2, -x5, x4, ...]

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        是什么：对 query 或 key 应用 RoPE。

        输入：  [batch, num_heads, seq_len, head_dim]
                x 可以是 Q 或 K（**不是** V——value 不需要位置）
        输出： 同形状，按位置相关角度旋转

        为什么只用于 Q 和 K：
        attention 分数 = Q_i · K_j 决定关注哪些 value。
        我们希望该分数依赖相对位置。
        V 向量承载内容——内容本身与位置无关。
        位置只影响**关注哪些** token。
        """
        # 是什么：取当前序列长度的 cos 和 sin
        # 为什么：若 seq_len=512 而 max_seq_len=2048，只需
        #      缓存 cos/sin 表的前 512 行。
        cos = self.cos_cached[:seq_len]   # [seq_len, head_dim]
        sin = self.sin_cached[:seq_len]   # [seq_len, head_dim]

        # 是什么：加 batch 和 head 维以便广播
        # 为什么：cos/sin 是 [seq_len, head_dim]。需与
        #      x [batch, heads, seq_len, head_dim] 相乘。
        #      unsqueeze(0).unsqueeze(0) 在位置 0、1 加维：
        #      [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        #      即可在 batch 和 head 上正确广播。
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # 是什么：执行旋转：x_rotated = x*cos(θ) + rotate_half(x)*sin(θ)
        # 为什么：数学上等价于对每对维度应用 2D 旋转矩阵，
        #      但用纯逐元素运算实现——更快、可并行。
        return (x * cos) + (self.rotate_half(x) * sin)
```

---

**上一章：** [第 3 章 — 嵌入](03_embeddings.md)
**下一章：** [第 5 章 — 注意力](05_attention.md)

