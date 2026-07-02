# 从零实现 Qwen3

本文件夹中的 [standalone-qwen3.ipynb](standalone-qwen3.ipynb) / [standalone-qwen3_ch.ipynb](standalone-qwen3_ch.ipynb) Jupyter notebook 包含 Qwen3 0.6B、1.7B、4B、8B 和 32B 的从零实现。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

本文件夹中的 [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb) / [standalone-qwen3-moe_ch.ipynb](standalone-qwen3-moe_ch.ipynb) 和 [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb) / [standalone-qwen3-moe-plus-kvcache_ch.ipynb](standalone-qwen3-moe-plus-kvcache_ch.ipynb) Jupyter notebook 包含 30B-A3B 混合专家（MoE）的从零实现，包括 Thinking、Instruct 和 Coder 模型变体。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-coder-flash-overview.webp?123" width="430px">

&nbsp;
# Qwen3 从零实现代码

本文件夹中的独立 notebook 以线性方式包含从零实现代码：

1. [standalone-qwen3.ipynb](standalone-qwen3.ipynb) / [standalone-qwen3_ch.ipynb](standalone-qwen3_ch.ipynb)：不带额外功能的稠密 Qwen3 模型
2. [standalone-qwen3-plus-kvcache.ipynb](standalone-qwen3-plus-kvcache.ipynb) / [standalone-qwen3-plus-kvcache_ch.ipynb](standalone-qwen3-plus-kvcache_ch.ipynb)：与上面相同，但带 KV 缓存以提升推理效率
3. [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb) / [standalone-qwen3-moe_ch.ipynb](standalone-qwen3-moe_ch.ipynb)：类似第一个 notebook，但是混合专家（MoE）变体
4. [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb) / [standalone-qwen3-moe-plus-kvcache_ch.ipynb](standalone-qwen3-moe-plus-kvcache_ch.ipynb)：与上面相同，但带 KV 缓存以提升推理效率

或者，我也将代码组织成了 Python 包，见 [这里](../../pkg/llms_from_scratch/)（包括单元测试和 CI），你可以按下面说明运行。

&nbsp;
# 训练

`Qwen3Model` 类的实现风格与 `GPTModel` 类类似，因此可以作为第 5 章训练以及第 6、7 章微调的直接替换。

&nbsp;
# 通过 `llms-from-scratch` 包使用 Qwen3

要便捷地使用 Qwen3 从零实现，你也可以使用基于本仓库 [pkg/llms_from_scratch](../../pkg/llms_from_scratch) 源代码的 `llms-from-scratch` PyPI 包。

&nbsp;
#### 1）安装

```bash
pip install llms_from_scratch tokenizers
```

&nbsp;
#### 2）模型和文本生成设置

指定要使用的模型：

```python
USE_REASONING_MODEL = True
# 若 USE_REASONING_MODEL = False，则使用 base 模型

USE_INSTRUCT_MODEL = False
# 若 USE_REASONING_MODEL = True 且 USE_INSTRUCT_MODEL = True，
# 则使用 instruct 模式（不含 reasoning）
# 若 USE_REASONING_MODEL = False，此设置无效


# 对 Qwen3 Coder Flash 模型同样使用
# USE_REASONING_MODEL = True
```

可由用户定义的基本文本生成设置。150 个 token 时，0.6B 模型约需 1.5 GB 内存。

```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3a）0.6B 模型的权重下载和加载

以下代码根据上面的模型选择（reasoning 或 base）自动下载权重文件。注意，本节聚焦 0.6B 模型。如果你想使用更大的模型（1.7B、4B、8B 或 32B），请跳过本节并继续 3b)。

```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"    
else:
    filename = "qwen3-0.6B-base.pth"   
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

然后按以下方式加载模型权重：

```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device);
```

&nbsp;
#### 3b）更大 Qwen 模型的权重下载和加载

如果你对更大的 Qwen 模型（例如 1.7B、4B、8B 或 32B）感兴趣，请使用以下代码替代 3a) 下的代码，这需要额外的代码依赖：

```bash
pip install safetensors huggingface_hub
```

然后使用以下代码（修改 `USE_MODEL` 以选择所需的模型大小）

```python
USE_MODEL = "1.7B"

if USE_MODEL == "1.7B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
elif USE_MODEL == "4B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
elif USE_MODEL == "8B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
elif USE_MODEL == "14B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
elif USE_MODEL == "32B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
elif USE_MODEL == "30B-A3B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_30B_A3B as QWEN3_CONFIG
else:
    raise ValueError("Invalid USE_MODEL name.")
    
repo_id = f"Qwen/Qwen3-{USE_MODEL}"
local_dir = f"Qwen3-{USE_MODEL}"

if not USE_REASONING_MODEL:
  repo_id = f"{repo_id}-Base"
  local_dir = f"{local_dir}-Base"
```

现在，下载权重并加载到 `model` 中：

```python
from llms_from_scratch.qwen3 import (
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen
)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

with device:
    model = Qwen3Model(QWEN3_CONFIG)

weights_dict = download_from_huggingface_from_snapshots(
    repo_id=repo_id,
    local_dir=local_dir
)
load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)  # only required for the MoE models
del weights_dict  # delete weight dictionary to free up disk space
```

&nbsp;
#### 4）初始化分词器

以下代码下载并初始化分词器：

```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    apply_chat_template=USE_REASONING_MODEL,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=not USE_INSTRUCT_MODEL
)
```

&nbsp;
#### 5）生成文本

最后，可以通过以下代码生成文本：

```python
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
```

```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"耗时: {total_time:.2f} 秒")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/秒")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"最大显存占用: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\n输出文本：\n\n", output_text + "...")
```

使用 Qwen3 0.6B reasoning 模型时，输出应类似于下方所示（在 A100 上运行）：

```
Time: 6.35 sec
25 tokens/sec
Max memory allocated: 1.49 GB


Output text:

 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems...
```

对于更大的模型，你可能更喜欢流式变体，它会在每个 token 生成后立即打印：

```python
from llms_from_scratch.generate import generate_text_simple_stream

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in generate_text_simple_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=150,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
```

&nbsp;
#### 专业技巧 1：使用编译加速推理

最多可获得约 4 倍加速，将

```python
model.to(device)
```

替换为

```python
model.to(device)
model = torch.compile(model)
```

注意：编译有显著的多分钟前期成本，加速效果在第一次 `generate` 调用后才会生效。

下表展示了 A100 上后续 `generate` 调用的性能对比：

|                          | 硬件            | Tokens/秒 | 显存     |
| ------------------------ | ----------------|----------- | -------- |
| Qwen3Model 0.6B          | Nvidia A100 GPU | 25         | 1.49 GB  |
| Qwen3Model 0.6B compiled | Nvidia A100 GPU | 107        | 1.99 GB  |

&nbsp;
#### 专业技巧 2：使用 KV 缓存加速推理

在 CPU 上运行模型时，你可以使用 KV 缓存版 `Qwen3Model` 直接替换以显著提升推理性能。（要了解更多关于 KV 缓存的内容，请参阅我的文章 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。）

```python
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Qwen3Model(QWEN_CONFIG_06_B)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

| Model           | 模式              | 硬件            | Tokens/秒 | GPU 显存 (VRAM) |
| --------------- | ----------------- | --------------- | ---------- | ----------------- |
| Qwen3Model 0.6B | Regular           | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model 0.6B | Regular compiled  | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model 0.6B | KV cache          | Mac Mini M4 CPU | 80         | -                 |
| Qwen3Model 0.6B | KV cache compiled | Mac Mini M4 CPU | 137        | -                 |
| Qwen3Model 0.6B | Regular           | Nvidia A100 GPU | 26         | 1.49 GB           |
| Qwen3Model 0.6B | Regular compiled  | Nvidia A100 GPU | 107        | 1.99 GB           |
| Qwen3Model 0.6B | KV cache          | Nvidia A100 GPU | 25         | 1.47 GB           |
| Qwen3Model 0.6B | KV cache compiled | Nvidia A100 GPU | 90         | 1.48 GB           |

注意，上述所有设置均已测试，可产生相同的文本输出。

&nbsp;
#### 专业技巧 3：批量推理

我们还可以通过批量推理进一步提升吞吐量。虽然这不是完全公平的对比（因为我们现在以更多输入序列运行推理），但这会以增加内存使用为代价提升 tokens/秒吞吐量。

这只需要对准备 prompt 的代码做小幅修改。例如，考虑下面的批量 prompt：

```python
from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B
# ...

prompts = [
    "Give me a short introduction to neural networks.",
    "Give me a short introduction to machine learning.",
    # ...
]

tokenized_prompts = [tokenizer.encode(p) for p in prompts]
max_len = max(len(t) for t in tokenized_prompts)
padded_token_ids = [
    t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
]
input_tensor = torch.tensor(padded_token_ids).to(device)

output_token_ids = generate_text_simple(
    model=model,
    idx=input_tensor,
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

KV 缓存版本的代码类似，但需要使用这些直接替换：

```python
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```

以下实验在 batch size 为 8 时运行。

| Model            | 模式              | 硬件            | Batch size | Tokens/秒 | GPU 显存 (VRAM) |
| ---------------- | ----------------- | --------------- | ---------- | ---------- | ----------------- |
| Qwen3Model  0.6B | Regular           | Nvidia A100 GPU | 8          | 184        | 2.19 GB           |
| Qwen3Model 0.6B  | Regular compiled  | Nvidia A100 GPU | 8          | 351        | 2.19 GB           |
| Qwen3Model 0.6B  | KV cache          | Nvidia A100 GPU | 8          | 140        | 3.13 GB           |
| Qwen3Model 0.6B  | KV cache compiled | Nvidia A100 GPU | 8          | 280        | 1.75 GB           |
