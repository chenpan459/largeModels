# 第 9 章 — 推理：让模型开口说话

## 生成与训练有何不同

| 方面 | 训练 | 推理（生成） |
|---|---|---|
| **目标** | 学会正确预测下一个 token | 实际生成新文本 |
| **输入** | 带标签的完整序列 | 仅 prompt（无标签） |
| **Teacher forcing** | 有 — 展示正确答案 | 无 — 模型自己生成后续内容 |
| **前向传播** | 整段序列一次前向 | 每生成一个新 token 一次前向 |
| **速度** | 快（batch 并行） | 慢（逐 token 顺序生成） |
| **Causal mask** | 防止看到未来 token | 未来 token 尚不存在 |
| **Dropout** | 开启（用于正则化） | 关闭（输出更确定） |
| **梯度** | 有（反向传播） | 无（不进行学习） |

## 朴素的生成循环

最简单的实现（每步都遍历全部 token）：

```python
for _ in range(max_new_tokens):
    # 每次都重新计算整段序列！← 非常浪费！
    logits, _ = model(input_ids)           # 处理全部 token
    next_token = sample(logits[:, -1, :])  # 只用最后一个位置的预测
    input_ids = torch.cat([input_ids, next_token], dim=1)
```

**问题：** 当我们要添加 token 501 时，token 500 已经被处理了 499 次！

## KV Cache — 最大的加速手段

### 核心洞察

在自回归生成过程中，先前算出的 Key 和 Value 不会改变。无论我们在预测 token 1 还是 token 500，token 0 的 K 和 V 都是相同的。

**没有 KV Cache：** 每步都为所有 token 重新计算 K 和 V。  
**有 KV Cache：** 只为新 token 计算 K、V，追加到 cache 中，复用旧的。

```
步骤 1: 处理 "The"           → 将 K["The"], V["The"] 存入 cache
步骤 2: 处理 "cat"           → 复用 "The" 的 K,V，计算 "cat" 的 K,V
步骤 3: 处理 "sat"           → 复用 "The","cat" 的 K,V，计算 "sat" 的
...
步骤 500: 处理 "mat"         → 复用 499 个 token 的 K,V，只计算 1 个新的
```

### 速度提升

| 序列长度 | 无 KV Cache | 有 KV Cache | 加速比 |
|---|---|---|---|
| 100 | 5,050 ops | 100 ops | 50× |
| 500 | 125,250 ops | 500 ops | 250× |
| 1000 | 500,500 ops | 1000 ops | 500× |
| 4096 | 8.3M ops | 4096 ops | **2048×** |

生成越长，KV cache 越重要！

### 内存开销

KV cache 存储 `2 * num_layers * num_heads * seq_len * head_dim` 个浮点数：

GPT-2 small 生成 1000 个 token 时：
```
2 × 12 × 12 × 1000 × 64 = 18,432,000 浮点数
= 18.4M × 4 字节 (float32) = 73.7 MB
= 18.4M × 2 字节 (bfloat16) = 36.8 MB
```

对小模型尚可接受，但对 GPT-3（96 层、96 头）× 4096 token：
```
2 × 96 × 96 × 4096 × 128 = 96.6 亿浮点数 = 38.6 GB!
```

这就是为什么长上下文推理需要巨大的 GPU 显存，或更省内存的 KV cache 技术。

### KV Cache 实现思路

```python
# 简化的 KV cache（概念示意 — 非完整实现）
class GPTWithKVCache(GPT):
    def generate_with_cache(self, input_ids, max_new_tokens):
        # 预填充（Prefill）：处理 prompt，存储 K,V
        kv_cache = []  # 每层一个 (K, V) 元组的列表
        
        # 第一次前向：处理完整 prompt
        logits, new_kv = self.forward_with_cache(input_ids, kv_cache=None)
        kv_cache = new_kv  # 存储以供复用
        
        for _ in range(max_new_tokens):
            next_token = sample(logits[:, -1, :])
            # 只对新 token 前向，复用缓存的 K,V
            logits, new_kv = self.forward_with_cache(
                next_token.unsqueeze(1),  # 只有 1 个新 token！
                kv_cache=kv_cache
            )
            kv_cache = new_kv  # 将新的 K,V 追加到 cache
            input_ids = torch.cat([input_ids, next_token], dim=1)
```

## 采样策略 — 如何选取下一个 token

### Greedy 采样（temperature = 0）

始终选取概率最高的那一个 token。

```
提示: "The cat sat on the"
Logits: [the: 9.2,  a: 8.1,  my: 3.2,  their: 1.1, ...]
                                ↑ 总是选这个
结果: "The cat sat on the mat. The cat sat on the mat. The cat..."  ← 重复！
```

**问题：** 确定性 → 同一 prompt 总是得到相同输出，容易重复。

### Temperature 采样

在 softmax 之前缩放 logits。温度越低，分布越尖锐（选择更自信）；温度越高，分布越平坦（更随机）。

```python
# Temperature 对玩具分布的影响：
logits = [2.0, 1.0, 0.5, 0.1]  # 4 个可能的 token

# T = 0.5（低温 — 更自信）：
scaled = [2.0/0.5, 1.0/0.5, 0.5/0.5, 0.1/0.5]  # → [4.0, 2.0, 1.0, 0.2]
probs  = softmax([4.0, 2.0, 1.0, 0.2])          # → [0.86, 0.12, 0.02, 0.00]
# Token 0 有 86% 概率 — 非常自信！

# T = 1.0（标准）：
probs = softmax([2.0, 1.0, 0.5, 0.1])           # → [0.56, 0.21, 0.13, 0.10]
# Token 0 = 56% — 分布较均衡

# T = 2.0（高温 — 更有创意）：
scaled = [2.0/2.0, 1.0/2.0, 0.5/2.0, 0.1/2.0]  # → [1.0, 0.5, 0.25, 0.05]
probs  = softmax([1.0, 0.5, 0.25, 0.05])         # → [0.36, 0.22, 0.22, 0.20]
# 更平坦 — token 0 仅 36%，token 2 和 3 也有竞争力
```

**同一 prompt，不同 temperature：**

```
T=0.2（聚焦）:  "The capital of France is Paris, which is located in the Île-de-France region."
T=0.8（均衡）:  "The capital of France is Paris, a city known for its art, cuisine, and the Eiffel Tower."
T=1.5（创意）:  "The capital of France is Paris, where baguettes dream of becoming croissants under moonlight."
```

### Top-K 采样

只考虑概率最高的 K 个 token，其余概率置为 0。

```
K=50: 只保留 top 50 token。常用默认值 — 过滤明显无意义 token。
K=10: 激进过滤。容易重复，但不会出现明显无意义 token。
K=1:  等同于 greedy（总是选 #1）。
```

### Top-P（Nucleus）采样

只考虑**最小**的 token 集合，使其累积概率超过 P。

```
按概率排序的 token: [0.45, 0.22, 0.13, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]

Top-P = 0.9:
  累积概率: 0.45+0.22+0.13+0.08+0.05 = 0.93 > 0.9
  保留前 5 个 token，丢弃其余。

Top-P = 0.5:
  累积概率: 0.45+0.22 = 0.67 > 0.5
  保留前 2 个 token。
```

**为何 Top-P 优于 Top-K？** Top-P 会随模型置信度自适应：
- 模型很确定时：保留少量 token（分布尖锐）
- 模型不确定时：保留更多 token（分布平坦）

Top-K 无论置信度如何，始终保留恰好 K 个 token。

### Beam Search

不是每次只选一个 token，而是维护多条「beam」（候选序列）：

```
Beam 宽度 = 3:

步骤 1: "The" → 3 个最佳下一 token：["cat"(0.3), "dog"(0.2), "man"(0.1)]
步骤 2: "The cat" → 3 个最佳续写：["sat"(0.4), "is"(0.2), "was"(0.15)]
        "The dog" → 3 个最佳：["ran"(0.35), "is"(0.2), "barked"(0.1)]
        "The man" → 3 个最佳：["walked"(0.3), "said"(0.25), "is"(0.1)]
        选出整体最优的 3 条序列：
        "The cat sat" (0.3×0.4=0.12), "The dog ran" (0.2×0.35=0.07), ...
```

**Beam search**：输出质量更高，但具有确定性（每次相同）且更慢。常用于翻译，而非创意写作。

### Repetition Penalty

生成过程中，对已出现过的 token 施加惩罚：

```
对每个候选 token：
  penalty = 1.0 若 token 不在近期历史中
  penalty = 0.5 若 token 近期出现过一次
  penalty = 0.2 若 token 近期多次出现

logits = logits * penalty
```

这可防止模型陷入循环：`"I like cats. I like cats. I like cats..."`

### 对比表

| 策略 | 随机性 | 质量 | 速度 | 适用场景 |
|---|---|---|---|---|
| **Greedy** (T=0) | 无 | 事实类任务较好 | 快 | 翻译、代码 |
| **Temperature** | 可控 | 因设置而异 | 快 | 创意写作 |
| **Top-K=50** | 低-中 | 常用默认 | 快 | 通用生成 |
| **Top-P=0.9** | 自适应 | 常用默认 | 快 | 聊天、对话 |
| **Beam Search** | 无 | 最佳 | 慢 3-5× | 翻译、摘要 |
| **T=0.7 + Top-P=0.9** | 中等 | 很好 | 快 | 🏆 推荐默认 |

## 完整推理代码

### 加载 Checkpoint

```python
import torch


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    是什么：从保存的 checkpoint 文件加载训练好的 GPT 模型。
    为什么：训练结束后会保存模型状态。要生成文本，
         需要把这些状态（权重、配置等）重新加载回来。
    """
    # 是什么：从磁盘加载 checkpoint 字典
    # 为什么：map_location 确保加载到正确设备（CPU/GPU）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 是什么：根据保存的配置重建模型
    model = GPT(checkpoint["config"])

    # 是什么：将训练好的权重加载进模型
    # 为什么：state_dict 包含训练期间学到的每个参数值
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)  # 移到 GPU
    model.eval()              # 推理时关闭 dropout

    print(f"已从 step {checkpoint['step']} 加载模型，"
          f"loss: {checkpoint['loss']:.4f}")
    return model
```

### 文本生成封装

```python
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: torch.device = None,
):
    """
    是什么：用训练好的 GPT 模型从 prompt 生成文本。
    为什么：高层接口 — tokenize → generate → decode。

    参数指南：
      temperature: 0.2 = 偏事实, 0.8 = 均衡, 1.5 = 奔放
      top_k:       50 = 标准, 10 = 保守, 0 = 关闭
      top_p:       0.9 = 推荐, 0.5 = 较窄, 1.0 = 关闭
    """
    device = device or next(model.parameters()).device

    # 是什么：将 prompt 字符串转为 token ID
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )

    # 是什么：运行自回归生成
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # 是什么：将生成的 token ID 转回字符串
    return tokenizer.decode(output_ids[0].tolist())
```

### 交互式生成示例

```python
# 加载训练好的模型
model = load_checkpoint("checkpoints/best_model.pt", device)

# 测试不同生成策略
prompts = [
    "Once upon a time, in a land far away,",
    "The secret to happiness is",
    "If I could travel anywhere in the world, I would go to",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"提示: {prompt}")
    print(f"{'='*60}")

    # 保守 — 适合事实类
    text = generate_text(
        model, tokenizer, prompt, temperature=0.3, top_k=20,
    )
    print(f"\n保守 (T=0.3, K=20):")
    print(f"  {text[:300]}")

    # 均衡 — 常用默认
    text = generate_text(
        model, tokenizer, prompt, temperature=0.8, top_k=50, top_p=0.9,
    )
    print(f"\n均衡 (T=0.8, K=50, P=0.9):")
    print(f"  {text[:300]}")

    # 创意 — 适合写作
    text = generate_text(
        model, tokenizer, prompt, temperature=1.3, top_k=100, top_p=0.95,
    )
    print(f"\n创意 (T=1.3, K=100, P=0.95):")
    print(f"  {text[:300]}")
```

---

**上一章：** [第 8 章 — 训练流程](08_training.md)
**下一章：** [第 10 章 — 完整脚本](10_full_script.md)
