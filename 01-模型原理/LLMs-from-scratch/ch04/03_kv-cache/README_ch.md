# 附加材料：KV 缓存



**此文件夹实现了向 GPT 模型添加 KV 缓存的功能。**

&nbsp;
## 概述

简而言之，KV 缓存存储中间的键（K）和值（V）计算结果以便在推理过程中重复使用，这会大幅提升生成响应时的速度。缺点是这会给代码增加一些复杂度、增加内存使用量，并且无法在训练期间使用。然而，在部署 LLM 时，推理速度的提升往往非常值得这些代码复杂度和内存方面的权衡。

&nbsp;
## 工作原理

假设 LLM 正在生成一些文本。具体来说，假设 LLM 得到了以下提示词："Time flies"。

下图展示了底层注意力分数计算的一个片段，使用了第 3 章中修改过的图示，并突出显示了键和值向量：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

现在，正如我们在第 2 章和第 4 章中学到的，LLM 一次生成一个单词（或 token）。假设 LLM 生成了单词 "fast"，因此下一轮的提示词变成了 "Time flies fast"。如下图所示：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

正如我们通过比较前两幅图可以看到的那样，前两个 token 的键和值向量完全相同，如果在每一轮下一个 token 的文本生成中重新计算它们将是一种浪费。

因此，KV 缓存的想法是实现一种缓存机制，存储先前生成的键和值向量以便重复使用，这有助于我们避免不必要的重复计算。

&nbsp;

## KV 缓存实现

实现 KV 缓存的方法有很多种，其主要思想都是我们在每个生成步骤中只计算新生成 token 的键和值张量。

我选择了一种强调代码可读性的简单实现方式。我认为直接浏览代码改动就能很容易地看出它是如何实现的。

此文件夹中有两个文件：

1. [`gpt_ch04.py`](gpt_ch04.py)：取自第 3 章和第 4 章的独立代码，用于实现 LLM 并运行简单的文本生成函数
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：与上面相同，但做了必要的修改以实现 KV 缓存。

你可以选择

a. 打开 [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) 文件，查找标记新改动的 `# NEW` 部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 使用你选择的文件差异对比工具查看这两个代码文件，比较其中的改动：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

为了总结实现细节，下面是一个简短的说明。

&nbsp;

### 1. 注册缓存缓冲区

在 `MultiHeadAttention` 构造函数内部，我们添加了两个缓冲区 `cache_k` 和 `cache_v`，用于保存跨步骤拼接起来的键和值：

```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)
```

&nbsp;

### 2. 带有 `use_cache` 标志的前向传播

接下来，我们扩展 `MultiHeadAttention` 类的 `forward` 方法，使其接受 `use_cache` 参数。在将新一批 token 投影为 `keys_new`、`values_new` 和 `queries` 之后，我们要么初始化 KV 缓存，要么追加到已有的缓存中：

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;


### 3. 清除缓存

在生成文本时，在各个独立的序列之间（例如多次文本生成调用之间），我们必须重置这两个缓冲区，因此我们还向 `MultiHeadAttention` 类添加了一个缓存重置方法：

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. 在完整模型中传播 `use_cache`

在完成对 `MultiHeadAttention` 类的修改之后，现在我们来修改 `GPTModel` 类。首先，我们在构造函数中为 token 索引添加位置跟踪：

```python
self.current_pos = 0
```

然后，我们将单行的块调用替换为一个显式循环，将 `use_cache` 传递给每个 transformer 块：

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上述改动还要求对 `TransformerBlock` 类做一个小修改，使其接受 `use_cache` 参数：
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，为了方便使用，我们为 `GPTModel` 添加一个模型级别的重置方法，一次性清除所有块的缓存：

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. 在生成中使用缓存

在完成对 `GPTModel`、`TransformerBlock` 和 `MultiHeadAttention` 的修改之后，下面来看看我们如何在一个简单的文本生成函数中使用 KV 缓存：

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

请注意，我们在步骤 c) 中，仅通过 `logits = model(next_idx, use_cache=True)` 向模型提供新的 token。而在不使用缓存的情况下，我们需要通过 `logits = model(idx[:, -ctx_len:], use_cache=False)` 向模型提供整个输入，因为它没有存储的键和值可供重用。

&nbsp;

## 简单的性能比较

在概念层面介绍完 KV 缓存之后，最大的问题是它在一个小例子的实际场景中表现如何。为了实际试用这个实现，我们可以将前面提到的两个代码文件作为 Python 脚本运行，它们会运行这个拥有 1.24 亿参数的小型 LLM，从一个 4-token 的提示词 "Hello, I am" 开始，生成 200 个新 token：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在配备 M4 芯片（CPU）的 Mac Mini 上，结果如下：

|                        | 每秒 token 数 |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

因此，正如我们所看到的，对于一个 1.24 亿参数的小模型和 200 token 的短序列长度，我们已经获得了约 5 倍的速度提升。（请注意，此实现是针对代码可读性进行优化的，而不是针对 CUDA 或 MPS 运行时速度进行优化，后者需要预先分配张量，而不是重新创建并拼接张量。）

**注意：** 模型在两种情况下都会生成“乱码”，即看起来像下面这样的文本：

> Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

这是因为我们还没有训练这个模型。下一章将会训练模型，之后你就可以在训练好的模型上使用 KV 缓存（不过，KV 缓存仅适用于推理阶段）来生成连贯的文本。这里我们使用未训练的模型，是为了让代码保持简单。

更重要的是，`gpt_ch04.py` 和 `gpt_with_kv_cache.py` 这两种实现生成的文本完全相同。这说明 KV 缓存的实现是正确的——因为很容易出现索引错误，从而导致结果出现分歧。


&nbsp;

## KV 缓存的优点和缺点

随着序列长度的增加，KV 缓存的好处和缺点会以以下方式变得更加明显：

- [优点] **计算效率提高**：如果没有缓存，第 *t* 步的注意力计算必须将新的查询与之前的 *t* 个键进行比较，因此累积计算量呈二次方增长，即 O(n²)。有了缓存，每个键和值只需计算一次，然后重复使用，这就把每一步的总复杂度降低为线性的 O(n)。

- [缺点] **内存使用量线性增长**：每个新 token 都会追加到 KV 缓存中。对于长序列和更大的 LLM，累积起来的 KV 缓存会变得越来越大，这可能会占用大量甚至过多的（GPU）内存。作为一种权宜之计，我们可以截断 KV 缓存，但这会带来更多的复杂性（不过，话说回来，在部署 LLM 时，这么做很可能仍然是值得的）。



&nbsp;
## 优化 KV 缓存实现

虽然我上面对 KV 缓存的概念性实现有助于清晰理解，并且主要是面向代码可读性和教学目的的，但要在实际场景中部署它（尤其是使用更大的模型和更长的序列长度时），还需要更细致的优化。

&nbsp;
### 扩展缓存规模时的常见陷阱

- **内存碎片化与重复分配**：如前所示，通过 `torch.cat` 不断拼接张量，会由于频繁的内存分配和重新分配而带来性能瓶颈。

- **内存使用量线性增长**：如果处理不当，对于非常长的序列，KV 缓存的大小会变得不切实际。

&nbsp;
#### 技巧 1：预分配内存

与其反复拼接张量，我们可以基于预期的最大序列长度，预先分配一个足够大的张量。这能确保内存使用保持一致，并减少开销。用伪代码表示，大致如下：

```python
# Example pre-allocation for keys and values
max_seq_len = 1024  # maximum expected sequence length
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

在推理过程中，我们随后就可以简单地将数据写入这些预分配张量的切片中。

&nbsp;
#### 技巧 2：通过滑动窗口截断缓存

为了避免撑爆 GPU 内存，我们可以实现一种带有动态截断功能的滑动窗口方法。通过滑动窗口，我们只在缓存中保留最后 `window_size` 个 token：


```python
# Sliding window cache implementation
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### 实践中的优化

你可以在 [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) 文件中找到这些优化。


在配备 M4 芯片（CPU）的 Mac Mini 上，使用 200-token 生成，并将窗口大小设置为等于上下文长度（以保证结果一致）的情况下，各代码的运行时间对比如下：

|                                  | 每秒 token 数 |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

不幸的是，这种速度优势在 CUDA 设备上消失了，因为这是一个非常小的模型，设备间的数据传输和通信开销超过了 KV 缓存为这种小模型带来的收益。


&nbsp;
## 附加资源

1. [从零实现 Qwen3 的 KV 缓存基准测试](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [从零实现 Llama 3 的 KV 缓存基准测试](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [理解并从零编写 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) —— 本 README 的更详细说明文章
