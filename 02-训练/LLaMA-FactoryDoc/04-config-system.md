# 04 — 配置系统

## 配置来源

LLaMA Factory 支持三种配置输入方式，优先级为：**代码传入 > CLI 覆盖 > YAML/JSON 文件**。

### 1. YAML 配置文件（推荐）

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

### 2. CLI 点号覆盖（OmegaConf dotlist）

```bash
llamafactory-cli train config.yaml learning_rate=1e-5 logging_steps=1
```

### 3. 纯 CLI 参数

```bash
llamafactory-cli train --model_name_or_path Qwen/Qwen3-4B --stage sft --finetuning_type lora
```

## 配置解析流程

`hparams/parser.py` 中的 `read_args()` 负责统一读取：

```python
def read_args(args=None):
    if args is not None:
        return args
    if sys.argv[1].endswith(".yaml"):
        override = OmegaConf.from_cli(sys.argv[2:])   # CLI 覆盖
        config = OmegaConf.load(sys.argv[1])          # YAML 文件
        return OmegaConf.to_container(OmegaConf.merge(config, override))
    elif sys.argv[1].endswith(".json"):
        # 同上，JSON 格式
    else:
        return sys.argv[1:]  # 纯 CLI 参数列表
```

解析后，`get_train_args()` / `get_infer_args()` / `get_eval_args()` 使用 HuggingFace 的 `HfArgumentParser` 将字典映射到 dataclass，并执行校验逻辑（设备兼容性、量化与微调方式冲突等）。

## YAML 配置结构

典型 SFT 配置分为以下区块（以 `examples/train_lora/qwen3_lora_sft.yaml` 为例）：

```yaml
### model
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: identity,alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048
max_samples: 1000
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen3-4b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: none  # none / wandb / tensorboard / swanlab / mlflow

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### eval
# eval_dataset: alpaca_en_demo
# val_size: 0.1
# eval_strategy: steps
# eval_steps: 500
```

## 五大参数 Dataclass

训练时 `get_train_args()` 返回五个 dataclass 的元组：

```python
model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
```

### ModelArguments（`model_args.py`）

| 参数 | 说明 |
|------|------|
| `model_name_or_path` | 基座模型路径或 Hub ID |
| `adapter_name_or_path` | LoRA 适配器路径（推理/继续训练） |
| `trust_remote_code` | 是否信任远程代码 |
| `quantization_bit` | 量化位数（4/8 bit QLoRA） |
| `quantization_method` | 量化方法：bitsandbytes / gptq / awq 等 |
| `infer_backend` | 推理后端：huggingface / vllm / sglang |
| `export_dir` | 模型导出目录 |
| `flash_attn` | FlashAttention 模式：auto / fa2 / sdpa |
| `use_unsloth` | 是否启用 Unsloth 加速 |
| `use_kt` | 是否启用 KTransformers |

### DataArguments（`data_args.py`）

| 参数 | 说明 |
|------|------|
| `dataset` | 数据集名称（逗号分隔，对应 dataset_info.json） |
| `eval_dataset` | 评测数据集 |
| `template` | 提示模板名称 |
| `cutoff_len` | 最大序列长度 |
| `max_samples` | 最大样本数（调试用） |
| `preprocessing_num_workers` | 预处理并行数 |
| `val_size` | 验证集比例 |
| `packing` | 是否启用序列打包 |
| `neat_packing` | 无交叉 attention 的打包模式 |
| `streaming` | 流式加载大数据集 |

### TrainingArguments（`training_args.py`）

扩展 HuggingFace `TrainingArguments`，额外支持 Ray 分布式参数。

| 参数 | 说明 |
|------|------|
| `output_dir` | 输出目录 |
| `per_device_train_batch_size` | 每卡 batch size |
| `gradient_accumulation_steps` | 梯度累积步数 |
| `learning_rate` | 学习率 |
| `num_train_epochs` | 训练轮数 |
| `bf16` / `fp16` | 混合精度 |
| `deepspeed` | DeepSpeed 配置文件路径 |
| `fsdp` | FSDP 配置 |
| `report_to` | 日志后端 |
| `resume_from_checkpoint` | 断点续训路径 |

### FinetuningArguments（`finetuning_args.py`）

| 参数 | 说明 |
|------|------|
| `stage` | 训练阶段：pt / sft / rm / ppo / dpo / kto |
| `finetuning_type` | 微调类型：lora / oft / freeze / full |
| `lora_rank` | LoRA 秩 |
| `lora_alpha` | LoRA 缩放系数 |
| `lora_target` | LoRA 目标模块（如 `all` 或 `q_proj,v_proj`） |
| `pref_loss` | DPO 损失：sigmoid / hinge / ipo / orpo / simpo |
| `use_dora` | 是否启用 DoRA |
| `pissa_init` | PiSSA 初始化 |
| `plot_loss` | 训练结束后绘制 loss 曲线 |
| `use_hyper_parallel` | 启用 HyperParallel FSDP2 |
| `use_mca` | 启用 Megatron-core 适配器 |

### GeneratingArguments（`generating_args.py`）

| 参数 | 说明 |
|------|------|
| `max_new_tokens` | 最大生成长度 |
| `temperature` | 采样温度 |
| `top_p` | nucleus 采样 |
| `top_k` | top-k 采样 |
| `do_sample` | 是否采样 |
| `repetition_penalty` | 重复惩罚 |

## 示例配置目录

| 目录 | 内容 |
|------|------|
| `examples/train_lora/` | LoRA 微调（SFT、DPO、KTO、预训练、奖励模型） |
| `examples/train_full/` | 全量微调 |
| `examples/train_qlora/` | QLoRA 量化微调 |
| `examples/inference/` | 推理配置 |
| `examples/merge_lora/` | LoRA 合并导出 |
| `examples/extras/` | 高级算法（GaLore、BAdam、OFT 等） |
| `examples/deepspeed/` | DeepSpeed ZeRO 0–3 + offload |
| `examples/accelerate/` | FSDP 配置 |
| `examples/ascend/` | 昇腾 NPU 配置 |
| `examples/v1/` | v1 实验架构配置 |

## 数据集注册表

`data/dataset_info.json` 是数据集名称到文件的映射：

```json
{
  "identity": {
    "file_name": "identity.json"
  },
  "alpaca_en_demo": {
    "file_name": "alpaca_en_demo.json"
  },
  "glaive_toolcall_en_demo": {
    "file_name": "glaive_toolcall_en_demo.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "tools": "tools"
    }
  },
  "mllm_demo": {
    "file_name": "mllm_demo.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

| 字段 | 说明 |
|------|------|
| `file_name` | 本地文件名 |
| `formatting` | 数据格式：alpaca / sharegpt |
| `columns` | 列名映射 |
| `tags` | ShareGPT 角色标签 |
| `hf_hub_url` | HuggingFace Hub 数据集 URL |
| `ms_hub_url` | ModelScope 数据集 URL |

自定义数据集：在 `data/` 目录添加 JSON 文件，并在 `dataset_info.json` 中注册即可。

## DeepSpeed / FSDP 配置

在训练 YAML 中引用：

```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
```

DeepSpeed 配置文件位于 `examples/deepspeed/`（ds_z0 到 ds_z3 + offload 变体）。

FSDP 配置位于 `examples/accelerate/fsdp_config.yaml`。

## Web UI 配置

LLaMA Board 将 UI 表单值保存为用户缓存目录下的 YAML（`webui/common.py` 的 `DEFAULT_CONFIG_DIR`），然后由 `Runner` 生成 `llamafactory-cli train ...` 命令并以子进程执行。
