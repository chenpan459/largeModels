# 09 - 模型转换与 GGUF 格式

## 1. 模块概述

llama.cpp 使用 **GGUF** (GGML Unified Format) 作为标准模型文件格式。转换流水线负责将 HuggingFace/PyTorch 模型转换为 GGUF，供 C++ 推理引擎加载。

涉及目录：
- `convert_hf_to_gguf.py` - 主转换入口
- `conversion/` - 各模型架构的转换逻辑
- `gguf-py/` - Python GGUF 读写库
- `convert_lora_to_gguf.py` - LoRA 转换
- `convert_llama_ggml_to_gguf.py` - 旧格式迁移

## 2. 转换流程

```
HuggingFace Model (PyTorch/SafeTensors)
    |
    v
convert_hf_to_gguf.py
    |
    +-- conversion/__init__.py    # HF class -> converter 映射
    +-- conversion/llama.py       # 架构特定转换
    +-- conversion/qwen.py
    +-- conversion/deepseek.py
    +-- ... (80+ 转换器)
    |
    v
gguf-py/gguf/                   # GGUF 文件写入
    |
    v
model.gguf (F16/F32)
    |
    v (可选)
llama-quantize                  # C++ 量化工具
    |
    v
model-q4_k_m.gguf (量化)
```

## 3. 主转换脚本

### 3.1 convert_hf_to_gguf.py

```bash
# 基本转换
python convert_hf_to_gguf.py /path/to/hf-model --outfile model.gguf

# 指定输出类型
python convert_hf_to_gguf.py /path/to/hf-model --outtype f16

# 列出支持的模型
python convert_hf_to_gguf.py --print-supported-models

# 分片输出
python convert_hf_to_gguf.py /path/to/hf-model --outfile model.gguf --split-max-size 2G
```

依赖：
- Python 3.10+
- PyTorch
- gguf-py (内嵌)
- transformers (可选, 部分模型)
- sentencepiece / tiktoken (分词器)

### 3.2 convert_lora_to_gguf.py

```bash
python convert_lora_to_gguf.py /path/to/lora --outfile lora.gguf \
    --base-model-id org/model-name
```

### 3.3 convert_hf_to_gguf_update.py

更新现有 GGUF 文件的 metadata（不重新转换权重）。

## 4. conversion/ 模块

### 4.1 架构映射

`conversion/__init__.py` 维护 HuggingFace model class 到 converter 的映射：

```python
TEXT_MODEL_MAP = {
    "LlamaForCausalLM": "llama",
    "Qwen2ForCausalLM": "qwen",
    "Qwen3ForCausalLM": "qwen",
    "DeepseekV3ForCausalLM": "deepseek",
    "Gemma3ForCausalLM": "gemma",
    "MistralForCausalLM": "llama",
    # ... 200+ 映射
}
```

### 4.2 转换器基类

`conversion/base.py`:

```python
class ModelBase:
    def set_gguf_parameters(self): ...    # 写入 metadata
    def modify_tensors(self, data, name, bid): ...  # 张量变换
    def prepare_tensors(self): ...         # 准备输出

class TextModel(ModelBase):
    # 文本模型基类

class MmprojModel(ModelBase):
    # 多模态投影模型
```

### 4.3 转换器文件 (80+)

| 文件 | 转换模型 |
|------|----------|
| `llama.py` | LLaMA, Mistral, Vicuna, Arcee |
| `qwen.py` | Qwen 1/2/3/3.5, Qwen2-VL |
| `qwen3vl.py` | Qwen3-VL |
| `deepseek.py` | DeepSeek V1/V2/V3/V3.2 |
| `gemma.py` | Gemma 1/2/3/3n/4 |
| `glm.py` | GLM-4, ChatGLM |
| `mamba.py` | Mamba, FalconMamba |
| `rwkv.py` | RWKV-6/7 |
| `bert.py` | BERT, ModernBERT, Jina |
| `llava.py` | LLaVA 系列 |
| `internvl.py` | InternVL |
| `granite.py` | IBM Granite |
| `phi.py` | Phi-2/3/3.5/MoE |
| `falcon.py` | Falcon, Falcon-H1 |
| `gpt2.py` | GPT-2 |
| `gptneox.py` | GPT-NeoX |
| `olmo.py` | OLMo/OLMo2/OLMoE |
| `mistral.py` | Mistral |
| `mistral3.py` | Mistral 3/4 Vision |
| `plamo.py` | PLaMo |
| `bitnet.py` | BitNet |
| `bloom.py` | BLOOM |
| `command_r.py` | Cohere Command-R |
| `dbrx.py` | DBRX |
| `exaone.py` | EXAONE |
| `hunyuan.py` | Hunyuan |
| `jais.py` | Jais |
| `jamba.py` | Jamba |
| `lfm2.py` | LFM2 |
| `llada.py` | LLaDA |
| `llama4.py` | LLaMA 4 |
| `gpt_oss.py` | gpt-oss (OpenAI MoE) |
| `nemotron.py` | Nemotron |
| `minicpm.py` | MiniCPM |
| `mimo.py` | Mimo |
| `dots1.py` | Dots1 |
| `dream.py` | Dream |
| `ernie.py` | ERNIE 4.5 |
| `kimi_linear.py` | Kimi Linear |
| `lighton_ocr.py` | LightOn OCR |
| `maincoder.py` | MainCoder |
| `mellum.py` | Mellum |
| `minimax.py` | MiniMax |
| `pangu.py` | Pangu |
| `pixtral.py` | Pixtral |
| `plm.py` | PLM |
| `refact.py` | Refact |
| `smolvlm.py` | SmolVLM |
| `stablelm.py` | StableLM |
| `starcoder.py` | StarCoder |
| `step3.py` | Step3 |
| `t5.py` | T5 |
| `talkie.py` | Talkie |
| `ultravox.py` | Ultravox |
| `wavtokenizer.py` | WavTokenizer |
| `xverse.py` | Xverse |
| `arctic.py` | Arctic |
| `afmoe.py` | AFMoE |
| `baichuan.py` | Baichuan |
| `bailingmoe.py` | BailingMoe |
| `chameleon.py` | Chameleon |
| `chatglm.py` | ChatGLM |
| `codeshell.py` | CodeShell |
| `cogvlm.py` | CogVLM |
| `deci.py` | DeciLM |
| `dotsocr.py` | DotsOCR |
| `falcon_h1.py` | Falcon-H1 |
| `grovemoe.py` | GroveMoE |
| `internlm.py` | InternLM |
| `januspro.py` | JanusPro |
| `kimivl.py` | KimiVL |
| `llava.py` | LLaVA |
| `qwenvl.py` | Qwen-VL |
| `sarashina2.py` | Sarashina2 |
| `smallthinker.py` | SmallThinker |
| `youtuvl.py` | YoutuVL |

## 5. gguf-py 库

**路径**: `gguf-py/`

Python GGUF 文件读写库，独立于 llama.cpp 使用：

```python
import gguf

# 写入
writer = gguf.GGUFWriter("model.gguf", "llama")
writer.add_uint32("llama.context_length", 8192)
writer.add_tensor("token_embd.weight", tensor_data)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()

# 读取
reader = gguf.GGUFReader("model.gguf")
for tensor in reader.tensors:
    print(tensor.name, tensor.shape)
```

目录结构：
```
gguf-py/
├── gguf/
│   ├── __init__.py
│   ├── gguf_reader.py
│   ├── gguf_writer.py
│   ├── constants.py       # GGML/GGUF 常量
│   ├── quants.py          # 量化函数
│   ├── lazy.py            # 延迟加载
│   └── vocab.py           # 分词器序列化
├── examples/
└── tests/
```

## 6. GGUF 文件格式

### 6.1 结构

```
+------------------+
| Magic: "GGUF"    |  4 bytes
| Version: uint32  |  4 bytes
| n_tensors: u64   |  8 bytes
| n_kv: u64        |  8 bytes
+------------------+
| KV Metadata      |  变长
|  general.architecture = "llama"
|  llama.context_length = 8192
|  tokenizer.ggml.model = "llama"
|  tokenizer.chat_template = "..."
|  ...             |
+------------------+
| Tensor Info      |  变长
|  name, dims,     |
|  type, offset    |
+------------------+
| Tensor Data      |  对齐到 32 字节
|  (raw bytes)     |
+------------------+
```

### 6.2 关键 Metadata 键

| 键 | 说明 |
|----|------|
| `general.architecture` | 架构名 (llama, qwen2, deepseek2, ...) |
| `general.file_type` | 量化类型 (F16=1, Q4_K_M=15, ...) |
| `llama.context_length` | 最大上下文长度 |
| `llama.embedding_length` | 隐藏层维度 |
| `llama.block_count` | Transformer 层数 |
| `llama.feed_forward_length` | FFN 中间维度 |
| `llama.attention.head_count` | 注意力头数 |
| `tokenizer.ggml.model` | 分词器类型 |
| `tokenizer.ggml.tokens` | 词表 |
| `tokenizer.chat_template` | Chat 模板 (Jinja2) |

### 6.3 张量命名规范

```
token_embd.weight          # Token embedding
blk.{N}.attn_norm.weight   # Layer N attention norm
blk.{N}.attn_q.weight      # Layer N Q projection
blk.{N}.attn_k.weight      # Layer N K projection
blk.{N}.attn_v.weight      # Layer N V projection
blk.{N}.attn_output.weight # Layer N output projection
blk.{N}.ffn_norm.weight    # Layer N FFN norm
blk.{N}.ffn_gate.weight    # Layer N FFN gate
blk.{N}.ffn_up.weight      # Layer N FFN up
blk.{N}.ffn_down.weight    # Layer N FFN down
output_norm.weight         # Output norm
output.weight              # Output projection
```

## 7. 完整转换示例

```bash
# 1. 安装依赖
pip install torch transformers sentencepiece

# 2. 下载 HF 模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen2.5-7b

# 3. 转换为 GGUF (F16)
python convert_hf_to_gguf.py ./qwen2.5-7b --outfile qwen2.5-7b-f16.gguf --outtype f16

# 4. (可选) 生成 importance matrix
./build/bin/llama-imatrix -m qwen2.5-7b-f16.gguf -f calibration.txt -o imatrix.dat -ngl 99

# 5. 量化
./build/bin/llama-quantize --imatrix imatrix.dat qwen2.5-7b-f16.gguf qwen2.5-7b-q4_k_m.gguf Q4_K_M

# 6. 推理
./build/bin/llama-cli -m qwen2.5-7b-q4_k_m.gguf -p "Hello"
```

## 8. 添加新模型转换器

1. 在 `conversion/` 创建 `xxx.py`，继承 `TextModel`
2. 实现 `set_gguf_parameters()` 和 `modify_tensors()`
3. 在 `conversion/__init__.py` 的 `TEXT_MODEL_MAP` 注册 HF class
4. 在 `get_model_class()` 中添加实例化逻辑
5. 测试: `python convert_hf_to_gguf.py /path/to/model --outfile test.gguf`
