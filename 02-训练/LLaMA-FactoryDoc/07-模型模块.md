# 07 — 模型模块

## 模块概览

模型模块位于 `src/llamafactory/model/`，负责模型/tokenizer 的加载、适配器注入和兼容性补丁。

```
loader.py → patcher.py → adapter.py → model_utils/
   ↓            ↓             ↓
加载权重    兼容性修复    LoRA/OFT/Full/Freeze
```

## 模型加载

### 统一入口

`model/loader.py` 提供两个核心函数：

```python
tokenizer_module = load_tokenizer(model_args)   # → {"tokenizer", "processor"}
model = load_model(tokenizer, model_args, finetuning_args, is_trainable)
```

### load_tokenizer 流程

```
1. try_download_model_from_other_hub()  # HF/ModelScope 镜像下载
2. AutoTokenizer.from_pretrained()
3. patch_tokenizer()                    # 修复特殊 token
4. AutoProcessor.from_pretrained()      # 多模态 processor（可选）
5. patch_processor()
```

### load_model 流程

```
1. AutoConfig.from_pretrained()
2. patch_config()                       # 量化、dtype、attention 配置
3. 选择 AutoModel 类:
   - AutoModelForCausalLM              # 文本 LLM
   - AutoModelForImageTextToText       # 多模态 VLM
   - AutoModelForSeq2SeqLM             # Seq2Seq
   - AutoModelForCausalLMWithValueHead # RM/PPO
4. from_pretrained() 加载权重
5. patch_model()                        # 模型兼容性修复
6. init_adapter()                       # LoRA/OFT/Full/Freeze
7. register_autoclass()                 # 自动类注册
```

## 适配器机制

`model/adapter.py` 的 `init_adapter()` 根据 `finetuning_type` 分派：

### Full Tuning

```python
def _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32):
    # 所有参数 requires_grad=True（排除 forbidden_modules）
    # 可训练参数转为 fp32 以保证精度
```

### Freeze Tuning

```python
def _setup_freeze_tuning(model, finetuning_args, ...):
    # 根据 freeze_trainable_layers 确定可训练层
    # 正数 = 最后 N 层，负数 = 最前 N 层
    # freeze_trainable_modules 指定模块名
```

### LoRA

```python
def _setup_lora_tuning(model, finetuning_args, ...):
    # 1. 查找目标模块（lora_target）
    # 2. 创建 LoraConfig(rank, alpha, target_modules, ...)
    # 3. get_peft_model() 注入 LoRA 层
    # 4. 支持：PiSSA 初始化、rsLoRA、DoRA、LoRA+
```

LoRA 目标模块查找逻辑（`model_utils/misc.py`）：

- `lora_target=all` → `find_all_linear_modules()` 自动发现所有 Linear 层
- 指定模块名 → 精确匹配
- 多模态模型 → `patch_target_modules()` 排除 vision tower

### OFT（Orthogonal Fine-Tuning）

通过 PEFT 的 `OFTConfig` 实现，参数效率介于 LoRA 和 Full 之间。

## 量化

`model/model_utils/quantization.py` 支持多种量化方法：

| 方法 | 说明 | 用途 |
|------|------|------|
| bitsandbytes (BNB) | 4/8 bit 动态量化 | QLoRA 训练 |
| GPTQ | 训练后量化 | 推理加速 |
| AWQ | 激活感知量化 | 推理加速 |
| AQLM | 加性量化 | 极低比特 |
| HQQ | 半二次量化 | 快速量化 |
| EETQ | 8 bit 量化 | 训练/推理 |
| FP8 | 8 bit 浮点 | H100 等新硬件 |

QLoRA 配置示例：

```yaml
quantization_bit: 4
quantization_method: bitsandbytes
```

## 模型补丁

`model/patcher.py` 在加载后对模型做兼容性修复：

| 函数 | 说明 |
|------|------|
| `patch_config()` | 设置 attn_implementation、RoPE scaling、量化配置 |
| `patch_model()` | 修复 forward 签名、启用 gradient checkpointing |
| `patch_tokenizer()` | 添加/修复 pad_token、eos_token |
| `patch_processor()` | 多模态 processor 配置 |
| `patch_valuehead_model()` | RM/PPO 的 value head 补丁 |

## model_utils 子模块

| 文件 | 功能 |
|------|------|
| `attention.py` | FlashAttention-2 / SDPA 配置 |
| `quantization.py` | 量化方法选择与配置 |
| `rope.py` | RoPE scaling（YaRN、PI 等） |
| `visual.py` | 多模态视觉塔处理 |
| `moe.py` | Mixture-of-Experts 模型支持 |
| `unsloth.py` | Unsloth 加速加载 |
| `liger_kernel.py` | Liger Kernel 融合算子 |
| `checkpointing.py` | Gradient checkpointing |
| `embedding.py` | 词表扩展 |
| `valuehead.py` | Value head 加载（RM/PPO） |
| `misc.py` | 模块查找、参数统计 |

## 支持的模型类型

加载时根据 config 自动选择模型类：

| 类型 | AutoModel 类 | 示例 |
|------|-------------|------|
| 文本 LLM | `AutoModelForCausalLM` | Qwen3, LLaMA, Mistral |
| 多模态 VLM | `AutoModelForImageTextToText` | Qwen3-VL, LLaVA |
| Seq2Seq | `AutoModelForSeq2SeqLM` | T5, BART |
| 带 Value Head | `AutoModelForCausalLMWithValueHead` | RM/PPO 训练 |

## 模型导出

导出流程（`train/tuner.py:export_model()`）：

```
1. load_model() 加载基座 + adapter
2. 如果是 PeftModel → merge_and_unload() 合并 LoRA
3. 转换 dtype（bf16/fp16/fp32）
4. model.save_pretrained(export_dir)
5. tokenizer.save_pretrained(export_dir)
```

导出配置示例（`examples/merge_lora/qwen3_lora_sft.yaml`）：

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
adapter_name_or_path: saves/qwen3-4b/lora/sft
template: qwen3_nothink
export_dir: exports/qwen3-4b-merged
export_size: 2
export_device: cpu
```

## 关键函数速查

| 函数 | 文件 | 说明 |
|------|------|------|
| `load_tokenizer()` | `loader.py` | 加载 tokenizer + processor |
| `load_model()` | `loader.py` | 加载模型 + 注入适配器 |
| `init_adapter()` | `adapter.py` | LoRA/OFT/Full/Freeze 初始化 |
| `patch_config()` | `patcher.py` | 配置级补丁 |
| `patch_model()` | `patcher.py` | 模型级补丁 |
| `find_all_linear_modules()` | `model_utils/misc.py` | 自动发现 Linear 层 |
| `apply_liger_kernel()` | `model_utils/liger_kernel.py` | Liger 融合算子 |
