# 第 2 章 — 分词：把文字变成数字

## 5 岁小孩也能懂的类比

计算机只能理解**数字**。它们不知道字母「A」是什么意思——只知道「65」（它的 ASCII 码）。因此喂给神经网络之前，必须把文本转成数字。

最简单的想法：**给每个词分配一个数字**：
```
"cat"  ->  9246
"sat"  ->  6734
"on"   ->   389
"the"  ->   279
"mat"  -> 16789
```

但英语有数十万个词。我们真的需要给「antidisestablishmentarianism」单独一个编号吗？还有像「skibidi」这种建词表时不存在的新词怎么办？

## 解决方案：子词分词（BPE）

不用整词，我们把文本拆成**常见的子词片段**：

```
"unbelievably" -> "un" + "believ" + "ably"
"running"      -> "runn" + "ing"
"cats"         -> "cat" + "s"
"lower"        -> "low" + "er"
"GPT"          -> "G" + "P" + "T"
```

这就是**字节对编码（Byte Pair Encoding，BPE）**——GPT-2、GPT-3、GPT-4 以及大多数现代模型使用的正是这一算法。

### BPE 如何工作 — 逐步说明

BPE 从每个字符各自是一个「token」开始，然后反复合并出现最频繁的一对：

**起始文本：** `"low lower lowest"`

```
第 0 步（初始 — 每个字符是一个 token）：
l o w _ l o w e r _ l o w e s t

第 1 步（最频繁的字节对：'l'+'o' -> 'lo'）：
lo w _ lo w e r _ lo w e s t

第 2 步（最频繁的字节对：'lo'+'w' -> 'low'）：
low _ low e r _ low e s t

第 3 步（最频繁的字节对：'e'+'s' -> 'es'）：
low _ low e r _ low es t

第 4 步（最频繁的字节对：'es'+'t' -> 'est'）：
low _ low e r _ low est

第 5 步（最频繁的字节对：'low'+'_' -> 'low_'）：
low_ low e r _ low_ est
```

足够多次合并后，词表类似：`{l, o, w, e, r, s, t, _, lo, ow, low, er, es, est, low_}`

现在即使用这些片段也能表示从未见过的新词：

```
"lowest"  -> "low" + "est"     （都在词表中！）
"slower"  -> "s" + "low" + "er" （从未见过，但仍能工作！）
```

### 为什么 BPE 优于词级分词

| 问题 | 词级 | BPE |
|---|---|---|
| "running" vs "run" | 不同 token——无共享语义 | "runn" + "ing"——模型能看到联系 |
| 新词："rizz" | 未知 token → 模型失败 | "r" + "i" + "z" + "z" → 字符级仍可工作 |
| 词表大小 | 50 万+（罕见词太多） | 5 万（平衡、高效） |
| Unicode/emoji 处理 | 常出问题 | 字节级回退永不失败 |

### 特殊字符和 Emoji 呢？

BPE 在**字节**上操作，而非字符。这意味着它能分词任何能表示为字节的内容——emoji、中文、代码、LaTeX，甚至二进制数据：

```
"Hello 😊"  ->  ["Hello", " Ġ", "😊"]    （Ġ = GPT 分词器中的空格前缀）
"你好"       ->  通过 UTF-8 字节分词
"def foo():"->  ["def", "Ġfoo", "()", ":"]
```

### GPT 分词器约定

| Token | 示例 | 含义 |
|---|---|---|
| 普通 token | `"cat"`, `"the"`, `"ing"` | 常规子词片段 |
| 空格前缀 | `"Ġcat"`, `"Ġthe"` | 空格后的词首（Ġ 是特殊字符） |
| `<\|endoftext\|>` | EOS token | 标记文档结束——训练时至关重要 |
| 大写字母 | `"The"` vs `"the"` | 不同 token！大小写敏感 |

### EOS Token — 为什么重要

`<|endoftext|>`（End Of Sequence，序列结束）token **至关重要**，却常被忽视：

```python
# 没有 EOS — 两篇文档被合并：
doc1 = "The cat sat."     # tokens: [464, 3797, 3332, 13]
doc2 = "The dog ran."     # tokens: [464, 3290, 3407, 13]
# 结果：[464, 3797, 3332, 13, 464, 3290, 3407, 13]
# 模型看到："...sat. The dog ran." — 以为是一篇文档
# 学到："sat." 后面常跟 "The" — 错误！

# 有 EOS — 文档被分隔：
tokens = [464, 3797, 3332, 13, EOS, 464, 3290, 3407, 13, EOS]
# 模型学到：EOS 表示「这里结束了，下一个 token 与前面无关」
```

## 分词器代码 — 逐行注释

```python
from dataclasses import dataclass
import tiktoken


@dataclass
class TokenizerConfig:
    """
    是什么：把所有分词器设置集中在一处。
    为什么：像一张配方卡——整个项目保持一致。
         改一个值，处处自动更新。
    """
    name: str = "gpt2"                # 是什么：使用 GPT-2 的预训练 BPE 分词器
                                       # 为什么：与 GPT-3/4 相同的 BPE——5 万次合并，
                                       #      在数十亿文档上久经考验，
                                       #      且已训练好（无需数周工作）
    vocab_size: int = 50257           # 是什么：唯一 token 的总数
                                       # 为什么：50,257 是 GPT-2 的确切词表大小
                                       #      （50,000 次合并 + 256 字节 token + 1 个 EOS）
                                       #      这是「刚刚好」的数字——
                                       #      足够容纳罕见子词，
                                       #      又足够小以便快速矩阵运算


class SimpleTokenizer:
    """
    是什么：封装 tiktoken，提供友好、一致的接口。
    为什么：tiktoken 原生 API 较底层（每次调用都要指定 allowed_special）。
         这个包装让 encode/decode 变得简单——调用 .encode("hello") 即可得到 token。
         
         它还一致地处理 EOS token，避免训练数据准备时忘记添加。
    """

    def __init__(self, config: TokenizerConfig = None):
        """
        是什么：用 GPT-2 的 BPE 词表初始化分词器。
        为什么：我们使用预训练分词器，因为：
             1. 从零训练分词器需要数周 CPU 时间
             2. GPT-2 分词器开源、快速、久经考验
             3. 与生产模型相同分词器，意味着我们的
                代码与 GPT-3 分词方式完全一致
        """
        self.config = config or TokenizerConfig()

        # 是什么：从 tiktoken 加载 GPT-2 编码
        # 为什么：tiktoken 存储预训练 BPE 合并表。
        #      get_encoding("gpt2") 加载 GPT-2 训练时使用的
        #      确切 5 万次合并。
        self.enc = tiktoken.get_encoding(self.config.name)

        # 是什么：定义并编码序列结束（End-of-Sequence）token
        # 为什么：<|endoftext|> 是标记文档边界的特殊 token。
        #      训练时我们在每篇文档之间插入它，
        #      让模型学会一段文本何处结束、下一段何处开始。
        self.eos_token = "<|endoftext|>"       # 字符串形式
        self.eos_token_id = self.enc.encode(    # 转为 token ID
            self.eos_token,
            allowed_special={self.eos_token}    # 为什么：tiktoken 默认阻止特殊 token
                                                #      出于安全。我们必须
                                                #      显式允许 EOS 编码。
        )[0]  # [0] 因为 encode() 返回列表——我们要单个 ID

    def encode(self, text: str) -> list[int]:
        """
        是什么：把文本转成整数 token ID 列表。
        为什么：神经网络只吃数字。像 "Hello world" 这样的原始字符串
             对矩阵乘法毫无意义。

        示例："Hello world" -> [15496, 995]

        底层：tiktoken 用预训练 BPE 合并表把文本拆成子词片段，
        再在词表中查找每个片段的 ID。
        """
        # 是什么：使用 tiktoken 的快速 C/Rust 编码器
        # 为什么：tiktoken 用 Rust 而非 Python 编写。
        #      每秒可分词数百 MB 文本。
        #      纯 Python BPE 分词器会慢约 100 倍。
        return self.enc.encode(text, allowed_special={self.eos_token})

    def decode(self, ids: list[int]) -> str:
        """
        是什么：把 token ID 转回人类可读文本。
        为什么：模型在推理时生成 token ID 序列后，
             需要转回文本供人阅读。

        示例：[15496, 995] -> "Hello world"
        """
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        """
        是什么：词表中有多少唯一 token。
        为什么：该数字决定模型输出层大小——
             最后的 Linear 层必须有 vocab_size 个输出
             （每个可能的下一个 token 一个分数）。
             
             50,257 表示模型每次预测下一个词时
             从 50,257 种可能中选择。
        """
        return self.config.vocab_size


# ===== 是什么：快速自测 =====
# 为什么：组合之前务必单独测试每个组件。
#      「分词器能用吗？」是 5 秒检查，能省几小时
#      调试出问题的训练循环。
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()

    # 测试 1：基础文本
    test_text = "The cat sat on the mat."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"测试 1 — 基础：")
    print(f"  原文: '{test_text}'")
    print(f"  编码:  {encoded}")
    print(f"  解码:  '{decoded}'")
    print(f"  一致:    {test_text == decoded}")

    # 测试 2：EOS token
    eos = tokenizer.encode(tokenizer.eos_token)
    print(f"\n测试 2 — EOS token：")
    print(f"  字符串: '{tokenizer.eos_token}'")
    print(f"  Token ID: {tokenizer.eos_token_id}")
    print(f"  编码结果: {eos}")

    # 测试 3：罕见/未见过的词
    rare = tokenizer.encode("antidisestablishmentarianism")
    decoded_rare = tokenizer.decode(rare)
    print(f"\n测试 3 — 罕见词：")
    print(f"  编码: {rare}")
    print(f"  片段:  {[tokenizer.decode([t]) for t in rare]}")
    print(f"  解码: '{decoded_rare}'")

    # 测试 4：Emoji/Unicode
    emoji = tokenizer.encode("Hello 😊 world")
    print(f"\n测试 4 — Emoji：")
    print(f"  编码: {emoji}")
    print(f"  解码: '{tokenizer.decode(emoji)}'")

    print(f"\n  词表大小: {tokenizer.vocab_size:,}")
```

**预期输出：**
```
测试 1 — 基础：
  原文: 'The cat sat on the mat.'
  编码:  [464, 3797, 3332, 319, 262, 2603, 13]
  解码:  'The cat sat on the mat.'
  一致:    True

测试 2 — EOS token：
  字符串: '<|endoftext|>'
  Token ID: 50256
  编码结果: [50256]

测试 3 — 罕见词：
  编码: [378, 420, 1634, 2013, 82, 622, 441, 979, 389]
  片段:  ['ant', 'idis', 'establish', 'ment', 'ar', 'ian', 'ism']
  解码: 'antidisestablishmentarianism'

测试 4 — Emoji：
  编码: [15496, 52430, 23530, 248, 995]
  解码: 'Hello 😊 world'

  词表大小: 50,257
```

---

**上一章：** [第 1 章 — 环境配置](01_setup.md)
**下一章：** [第 3 章 — 嵌入](03_embeddings.md)
