# 将 GPT 转换为 Llama

本文件夹包含将第 4、5 章的 GPT 实现逐步转换为 Meta AI Llama 架构的代码，建议按以下顺序阅读：

- [converting-gpt-to-llama2_ch.ipynb](converting-gpt-to-llama2_ch.ipynb)：逐步将 GPT 转换为 Llama 2 7B，并加载 Meta AI 预训练权重
- [converting-llama2-to-llama3_ch.ipynb](converting-llama2-to-llama3_ch.ipynb)：将 Llama 2 模型转换为 Llama 3、Llama 3.1 和 Llama 3.2
- [standalone-llama32_ch.ipynb](standalone-llama32_ch.ipynb)：独立实现 Llama 3.2 的 notebook

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/gpt-and-all-llamas.webp">

&nbsp;
### 通过 `llms-from-scratch` 包使用 Llama 3.2

若想便捷使用 Llama 3.2 1B 和 3B 模型，也可使用本仓库 [pkg/llms_from_scratch](../../pkg/llms_from_scratch) 中的 `llms-from-scratch` PyPI 包。

&nbsp;
#### 1）安装

```bash
pip install llms_from_scratch blobfile
```

（加载分词器需要 `blobfile`。）

&nbsp;
#### 2）模型与文本生成设置

指定要使用的模型：

```python
MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"
```

可自定义的基本文本生成设置。注意：推荐的 8192 token 上下文在文本生成示例中约需 3 GB 显存。

```python
# 文本生成设置
if "instruct" in MODEL_FILE:
    PROMPT = "What do llamas eat?"
else:
    PROMPT = "Llamas eat"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3）下载与加载权重

根据上面选择的模型自动下载权重文件：

```python
import os
import requests

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"

if not os.path.exists(MODEL_FILE):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(MODEL_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"已下载至 {MODEL_FILE}")
```

然后按如下方式加载模型权重：

```python
import torch
from llms_from_scratch.llama3 import Llama3Model

if "1B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
elif "3B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
else:
    raise ValueError("Incorrect model file name")

model = Llama3Model(LLAMA32_CONFIG)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)
```

&nbsp;
#### 4）初始化分词器

以下代码下载并初始化分词器：

```python
from llms_from_scratch.llama3 import Llama3Tokenizer, ChatFormat, clean_text

TOKENIZER_FILE = "tokenizer.model"

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"

if not os.path.exists(TOKENIZER_FILE):
    urllib.request.urlretrieve(url, TOKENIZER_FILE)
    print(f"已下载至 {TOKENIZER_FILE}")
    
tokenizer = Llama3Tokenizer("tokenizer.model")

if "instruct" in MODEL_FILE:
    tokenizer = ChatFormat(tokenizer)
```

&nbsp;
#### 5）生成文本

最后，用以下代码生成文本：

```python
import time

from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

torch.manual_seed(123)

start = time.time()

token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
    top_k=TOP_K,
    temperature=TEMPERATURE
)

total_time = time.time() - start
print(f"耗时: {total_time:.2f} 秒")
print(f"{int(len(token_ids[0])/total_time)} tokens/秒")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"最大显存占用: {max_mem_gb:.2f} GB")

output_text = token_ids_to_text(token_ids, tokenizer)

if "instruct" in MODEL_FILE:
    output_text = clean_text(output_text)

print("\n\n输出文本：\n\n", output_text)
```

使用 Llama 3.2 1B Instruct 模型时，输出大致如下：

```
Time: 3.17 sec
50 tokens/sec
Max memory allocated: 2.91 GB


Output text:

 Llamas are herbivores, which means they primarily eat plants. Their diet consists mainly of:

1. Grasses: Llamas love to graze on various types of grasses, including tall grasses and grassy meadows.
2. Hay: Llamas also eat hay, which is a dry, compressed form of grass or other plants.
3. Alfalfa: Alfalfa is a legume that is commonly used as a hay substitute in llama feed.
4. Other plants: Llamas will also eat other plants, such as clover, dandelions, and wild grasses.

It's worth noting that the specific diet of llamas can vary depending on factors such as the breed,
```

&nbsp;
#### 技巧 1：用 FlashAttention 加速推理

可将 `Llama3Model` 直接替换为 `Llama3ModelFast`。详见 [pkg/llms_from_scratch/llama3.py](../../pkg/llms_from_scratch/llama3.py)。

`Llama3ModelFast` 在 `GroupedQueryAttention` 模块中用 PyTorch 的 `scaled_dot_product` 替代从零实现的缩放点积，在 Ampere 及更新 GPU 上会使用 `FlashAttention`。

A100 上的性能对比如下：

|                 | Tokens/秒 | 显存    |
| --------------- | ---------- | ------- |
| Llama3Model     | 42         | 2.91 GB |
| Llama3ModelFast | 54         | 2.91 GB |

&nbsp;
#### 技巧 2：用编译加速推理

最高约 4× 加速：将

```python
model.to(device)
```

替换为

```python
model = torch.compile(model)
model.to(device)
```

注意：编译有显著的多分钟前期开销，加速在第一次 `generate` 调用后生效。

A100 上后续 `generate` 调用的性能对比：

|                 | Tokens/秒 | 显存    |
| --------------- | ---------- | ------- |
| Llama3Model     | 170        | 3.12 GB |
| Llama3ModelFast | 177        | 3.61 GB |

&nbsp;
#### 技巧 3：用 KV cache 加速推理

在 CPU 上运行时，可使用带 KV cache 的 `Llama3Model` 替代实现显著加速。（KV cache 原理见 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。）

```python
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Llama3Model(LLAMA32_CONFIG)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
)
```

峰值显存仅列出 Nvidia CUDA 设备（便于计算）。其他设备显存占用可能相近（精度格式类似）；KV cache 在生成 150 token 时显存更低（但不同设备的矩阵乘法实现可能不同，峰值显存会有差异；更长上下文时 KV cache 显存可能急剧增长）。

| 模型        | 模式              | 硬件            | Tokens/秒 | GPU 显存 (VRAM) |
| ----------- | ----------------- | --------------- | ---------- | ----------------- |
| Llama3Model | Regular           | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | Regular compiled  | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | KV cache          | Mac Mini M4 CPU | 68         | -                 |
| Llama3Model | KV cache compiled | Mac Mini M4 CPU | 86         | -                 |
|             |                   |                 |            |                   |
| Llama3Model | Regular           | Mac Mini M4 GPU | 15         | -                 |
| Llama3Model | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Llama3Model | KV cache          | Mac Mini M4 GPU | 62         | -                 |
| Llama3Model | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|             |                   |                 |            |                   |
| Llama3Model | Regular           | Nvidia A100 GPU | 42         | 2.91 GB           |
| Llama3Model | Regular compiled  | Nvidia A100 GPU | 170        | 3.12 GB           |
| Llama3Model | KV cache          | Nvidia A100 GPU | 58         | 2.87 GB           |
| Llama3Model | KV cache compiled | Nvidia A100 GPU | 161        | 3.61 GB           |

以上设置均已验证可产生相同文本输出。
