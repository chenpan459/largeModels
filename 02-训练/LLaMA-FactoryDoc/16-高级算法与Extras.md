# 16 — 高级算法与 Extras

本文梳理 `0.9.6.dev0` 中不属于基础 full/freeze/LoRA 流程的优化器、adapter 初始化、损失与 kernel 加速，并给出“配置字段 → 参数定义 → 注入点 → 示例/依赖”的追踪方法。

## 1. 配置的三类来源

高级能力分散在三组参数中：

| 参数类 | 路径 | 代表功能 |
|---|---|---|
| `FinetuningArguments` | `hparams/finetuning_args.py` | GaLore、APOLLO、BAdam、Muon、LoRA+、PiSSA、DFT/ASFT/EAFT |
| `ModelArguments` | `hparams/model_args.py` | Unsloth、Liger、attention、量化、KT |
| `TrainingArguments` | `hparams/training_args.py` | FP8、Ray、Profiler、HF optimizer |

实际注入点：

- optimizer：`train/trainer_utils.py::create_custom_optimizer()`；
- adapter：`model/adapter.py::init_adapter()`；
- kernel/model patch：`model/patcher.py` 与 `model/model_utils/`；
- SFT loss：`train/sft/trainer.py::CustomSeq2SeqTrainer`；
- FP8：`train/fp8_utils.py`。

WebUI 只暴露部分字段；源码 dataclass 与 YAML 才是完整配置面。

## 2. 自定义 optimizer 总路由

`create_custom_optimizer()` 按固定优先级返回第一个命中的实现：

```text
use_galore
→ use_apollo
→ loraplus_lr_ratio is not None
→ use_badam
→ use_adam_mini
→ use_muon
→ None（交回 HF Trainer 默认 optimizer）
```

`FinetuningArguments.__post_init__()` 进一步禁止：

- LoRA 与 GaLore/APOLLO/BAdam 同时使用；
- GaLore、APOLLO、BAdam 三者同时启用多个。

因此不要依赖优先级“叠加算法”；大部分组合本来就不合法。

## 3. GaLore

目标：对梯度做低秩投影，降低全参训练 optimizer state/梯度成本。

配置来源 `GaloreArguments`：

```yaml
finetuning_type: full
use_galore: true
galore_target: all
galore_rank: 16
galore_update_interval: 200
galore_scale: 2.0
galore_proj_type: std
galore_layerwise: false
```

实现 `trainer_utils._create_galore_optimizer()`：

1. `galore_target=all` 时通过 `find_all_linear_modules()` 找 Linear；
2. 只收集 `requires_grad` 且二维以上的参数；
3. 为目标参数组加入 `rank/update_proj_gap/scale/proj_type`；
4. 根据 HF `optim` 选择 `GaLoreAdamW`、8-bit GaLore 或 `GaLoreAdafactor`；
5. layerwise 模式给每个参数创建 optimizer，并注册 `post_accumulate_grad_hook`。

layerwise 不支持 gradient accumulation（必须为 1），日志中的 gradient norm 会显示为 0。依赖：`requirements/galore.txt`。示例：`examples/extras/galore/llama3_full_sft.yaml`。

## 4. APOLLO

APOLLO 与 GaLore 共用低秩梯度思想，但带随机/SVD 投影和 norm-growth scaling。

参数 `ApolloArguments`：

- `apollo_target=all`
- `apollo_rank=16`
- `apollo_update_interval=200`
- `apollo_scale=32.0`
- `apollo_proj=svd|random`
- `apollo_proj_type=std|right|left`
- `apollo_scale_type=channel|tensor`
- `apollo_layerwise`
- `apollo_scale_front`

`_create_apollo_optimizer()` 只接受 `optim=adamw_torch`，创建 `APOLLOAdamW`。layerwise 同样要求 accumulation=1。依赖：`requirements/apollo.txt`；示例：`examples/extras/apollo/llama3_full_sft.yaml`。

## 5. BAdam

BAdam 只更新部分参数，提供两种模式：

### 5.1 Layer-wise

`badam_mode=layer` 使用外部 `badam.BlockOptimizer`：

- `badam_start_block`
- `badam_switch_mode=ascending|descending|random|fixed`
- `badam_switch_interval=50`，`-1` 不切换；
- 能感知 DeepSpeed ZeRO-3。

### 5.2 Ratio-wise

`badam_mode=ratio` 使用 `BlockOptimizerRatio`：

- `badam_update_ratio=0.05`
- `badam_mask_mode=adjacent|scatter`
- 不包含 embedding。

各 Trainer 在初始化时还安装 `BAdamCallback`，并替换 Accelerate 的旧版 clip-grad 实现。BAdam 会让可训练参数保持 half precision，见 `model/adapter.py::init_adapter()`。依赖：`requirements/badam.txt`；示例：`examples/extras/badam/llama3_full_sft.yaml`。

## 6. Muon 与 Adam-mini

### 6.1 Muon

`use_muon=true` 调 `_create_muon_optimizer()`，实现位于 `src/llamafactory/third_party/muon.py`：

- 二维、且名称不含 `embed/lm_head` 的参数进入 Muon；
- embedding、head、bias 等进入 AdamW 参数组；
- 学习率、weight decay、Adam betas/epsilon 复用 TrainingArguments。

示例：`examples/extras/muon/qwen2_full_sft.yaml`。Muon 是仓库内第三方实现，不需要 `requirements/muon.txt`。

### 6.2 Adam-mini

`use_adam_mini=true` 调外部 `Adam_mini`，从 model config 读取 hidden size、Q heads、KV heads，并获知是否 FSDP/ZeRO-3 分片。依赖：`requirements/adam-mini.txt`；示例：`examples/extras/adam_mini/qwen2_full_sft.yaml`。

## 7. LoRA+、rsLoRA、DoRA 与 PiSSA

这些能力都建立在 `finetuning_type=lora` 上。

### 7.1 LoRA+

字段：

```yaml
loraplus_lr_ratio: 16.0
loraplus_lr_embedding: 1.0e-6
```

`_create_loraplus_optimizer()` 把参数分为：

- LoRA A：基础学习率；
- LoRA B：`learning_rate * loraplus_lr_ratio`；
- LoRA B no-decay；
- embedding B：单独 `loraplus_lr_embedding`。

示例：`examples/extras/loraplus/llama3_lora_sft.yaml`。

### 7.2 rsLoRA 与 DoRA

- `use_rslora` 直接传 PEFT `LoraConfig.use_rslora`；
- `use_dora` 传 `use_dora`，但不兼容除 bitsandbytes 以外的 PTQ 模型。

它们没有独立 optimizer，核心注入在 `model/adapter.py::_setup_lora_tuning()`。

### 7.3 PiSSA

字段：

- `pissa_init`
- `pissa_iter=16`；设 `-1` 使用完整 PiSSA，否则 `pissa_niter_N` FSVD；
- `pissa_convert`：训练结束由 `PissaConvertCallback` 转成标准 LoRA。

adapter 层把 `init_lora_weights` 设置为 `pissa` 或 `pissa_niter_*`。限制：

- 只能用于 LoRA；
- 不能直接在量化模型上初始化；
- PPO、KTO 或需要 reference model 的 DPO 禁用；
- `scripts/pissa_init.py` 提供离线初始化/量化路径。

示例：`examples/extras/pissa/llama3_lora_sft.yaml`。

## 8. Unsloth

字段位于 `ModelArguments`：

- `use_unsloth`
- `use_unsloth_gc`

链路：

```text
model/loader.py
  → model/model_utils/unsloth.py 加载优化模型
model/adapter.py
  → get_unsloth_peft_model()/load_unsloth_peft_model()
model/model_utils/checkpointing.py
  → Unsloth gradient checkpointing
```

限制：

- adapter 只允许一个；
- OFT 不支持；
- 模型/版本不受支持时可能 fallback 到标准 Transformers/PEFT 路径；
- WebUI booster 的 `unsloth` 映射到 `use_unsloth=True`。

Unsloth 是模型装载与训练 patch，不是一个 optimizer，也不应与“4-bit”画等号；是否量化仍由 `quantization_method/bit` 控制。

## 9. Liger Kernel

v0 字段：`ModelArguments.enable_liger_kernel`。`model/model_utils/liger_kernel.py` 根据模型类型调用 Liger 的 `apply_liger_kernel_to_*`，替换 RoPE、RMSNorm、SwiGLU、CrossEntropy 等融合实现。

依赖：`requirements/liger-kernel.txt`（`liger-kernel>=0.6.3`）。WebUI booster `liger_kernel` 映射到该字段。

v1 不是复用 v0 helper，而是：

```text
ModelEngine
  → kernel_config.name == "liger_kernel"
  → KernelPlugin
  → plugins/model_plugins/kernels/liger_kernel_ops.py
```

v1 SFT 为 loss weights 需要 logits，因此创建 plugin 时传 `require_logits=is_train`，避免 fused linear CE 吃掉 logits。示例：`examples/v1/train_full/train_full_liger_kernel.yaml`。

## 10. FP8

参数 `TrainingArguments/Fp8Arguments`：

```yaml
fp8: true
fp8_backend: auto  # auto|torchao|te|msamp
fp8_enable_fsdp_float8_all_gather: false
```

`CustomSeq2SeqTrainer` 和 PT Trainer 在初始化前调用：

```text
configure_fp8_environment()
→ ACCELERATE_MIXED_PRECISION=fp8
→ 可选 FP8_BACKEND
→ create_fp8_kwargs()
```

TorchAO：

- `Float8LinearConfig.from_recipe_name("rowwise")`；
- 只转换二维 Linear；
- 跳过 embedding/lm_head/output/classifier；
- 跳过输入或输出维度不是 16 倍数的层。

Transformer Engine：

- 使用 HYBRID format、amax history 16；
- 因 HF Trainer 不传 kwargs handlers，源码 patch `Accelerator.__init__()` 注入 recipe。

依赖：

- `requirements/fp8.txt`：TorchAO；
- `requirements/fp8-te.txt`：Transformer Engine。

示例：

- `examples/extras/fp8/llama3_fp8_deepspeed_sft.yaml`
- `examples/extras/fp8/llama3_fp8_fsdp_sft.yaml`

源码说明要求 PyTorch 2.7+、Hopper GPU。FP8 与 4/8-bit weight-only QLoRA 是不同维度；parser 禁止 FP8 与模型量化同时启用。

## 11. DFT、ASFT 与 EAFT 损失

字段在 `FinetuningArguments`：

- `use_dft_loss`
- `use_asft_loss`、`asft_alpha=0.1`
- `use_eaft_loss`、`eaft_alpha=1.0`

`train/sft/trainer.py` 按 DFT → EAFT → ASFT 顺序设置 `compute_loss_func`：

- DFT：`trainer_utils.dft_loss_func`；
- EAFT：`eaft_loss_func(..., eaft_alpha)`；
- ASFT：需要 reference model，workflow 调 `create_ref_model()`，Trainer 前向时额外计算 `ref_logits`。

示例：

- `examples/extras/dft/qwen2_full_sft.yaml`
- `examples/extras/asft/llama2_full_asft.yaml`
- `examples/extras/asft/qwen2_full_asft.yaml`
- `examples/extras/eaft/qwen25_05b_eaft_full.yaml`

它们是 SFT token loss 的替代/重加权，不是 DPO 的 `pref_loss`。

## 12. OFT/QOFT、LoftQ、LLaMA Pro

### OFT/QOFT

`finetuning_type=oft` 使用 PEFT `OFTConfig`，字段包括 `oft_rank`、`oft_block_size`、`oft_target`、`module_dropout`。量化模型可做 OFT，因此有 `examples/extras/qoft/`。Unsloth 当前不支持 OFT。

### LoftQ

`scripts/loftq_init.py` 离线生成量化感知 LoRA 初始化；`examples/train_qlora/qwen3_lora_sft_otfq.yaml` 文件名保留为 `otfq`，使用时应核对 YAML 内容，不要仅凭文件名推断算法。

### LLaMA Pro

`scripts/llama_pro.py` 先扩展模型层；训练时 `use_llama_pro` 让 freeze/LoRA 只作用于扩展 blocks。全参训练禁止该开关。示例：`examples/extras/llama_pro/llama3_freeze_sft.yaml`。

## 13. 组合与排错

| 组合 | 状态 |
|---|---|
| LoRA + LoRA+/rsLoRA/DoRA | 支持，具体组合仍需 PEFT 验证 |
| LoRA + PiSSA | 支持初始化；对齐阶段有额外限制 |
| LoRA + GaLore/APOLLO/BAdam | 参数校验禁止 |
| GaLore + APOLLO/BAdam | 参数校验禁止 |
| FP8 + 4/8-bit quantization | parser 禁止 |
| Unsloth + OFT | 禁止 |
| KT + 非 LoRA | 禁止 |
| layerwise GaLore/APOLLO + gradient accumulation | 禁止 |

排错顺序：

1. 看 `hparams/*_args.py` 字段是否属于当前架构；
2. 看 `__post_init__()` 是否拒绝组合；
3. 看 `trainer_utils.create_custom_optimizer()` 是否真正命中；
4. 看 requirements 是否安装；
5. 从 `examples/extras/` 的最小配置开始；
6. 检查日志中的 “Using ... optimizer/kernel” 而非只相信 YAML。

## 14. 源码与配置地图

- 算法参数：`src/llamafactory/hparams/finetuning_args.py`
- 模型加速参数：`src/llamafactory/hparams/model_args.py`
- FP8 参数：`src/llamafactory/hparams/training_args.py`
- optimizer/loss：`src/llamafactory/train/trainer_utils.py`
- SFT loss 注入：`src/llamafactory/train/sft/trainer.py`
- adapter：`src/llamafactory/model/adapter.py`
- Unsloth：`src/llamafactory/model/model_utils/unsloth.py`
- Liger：`src/llamafactory/model/model_utils/liger_kernel.py`
- FP8：`src/llamafactory/train/fp8_utils.py`
- Muon：`src/llamafactory/third_party/muon.py`
- 配置示例：`examples/extras/`
- 可选依赖：`requirements/`
