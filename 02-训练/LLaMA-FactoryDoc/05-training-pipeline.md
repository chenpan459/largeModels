# 05 — 训练流水线

## 训练编排器

核心入口：`src/llamafactory/train/tuner.py`

```python
def run_exp(args=None, callbacks=None):
    args = read_args(args)
    ray_args = get_ray_args(args)
    if ray_args.use_ray:
        _ray_training_function(...)   # Ray 集群
    else:
        _training_function(...)     # 本地/分布式
```

`_training_function()` 解析参数、注册回调后，按 `finetuning_args.stage` 路由到对应 workflow。

## Stage 路由

```python
# tuner.py 核心路由逻辑
if finetuning_args.use_hyper_parallel:
    run_{pt,sft}_hp(...)       # HyperParallel FSDP2
elif finetuning_args.use_mca:
    run_{pt,sft,dpo}_mca(...)  # Megatron-core
elif finetuning_args.stage == "pt":
    run_pt(...)
elif finetuning_args.stage == "sft":
    run_sft(...)
elif finetuning_args.stage == "rm":
    run_rm(...)
elif finetuning_args.stage == "ppo":
    run_ppo(...)
elif finetuning_args.stage == "dpo":
    run_dpo(...)
elif finetuning_args.stage == "kto":
    run_kto(...)
```

## 训练阶段

| Stage | 模块 | 说明 | 典型用途 |
|-------|------|------|---------|
| `pt` | `train/pt/` | 继续预训练 | 领域语料预训练 |
| `sft` | `train/sft/` | 监督微调 | 指令跟随、对话 |
| `rm` | `train/rm/` | 奖励模型 | RLHF 偏好打分 |
| `ppo` | `train/ppo/` | PPO 强化学习 | RLHF 策略优化 |
| `dpo` | `train/dpo/` | 直接偏好优化 | 对齐训练（无需 RM） |
| `kto` | `train/kto/` | KTO 优化 | 二元反馈对齐 |

每个 stage 目录结构一致：

```
train/{stage}/
├── workflow.py    # 编排：数据 → 模型 → Trainer → 保存
└── trainer.py     # 自定义 Trainer（继承 HF Trainer / TRL）
```

## SFT 工作流详解

`train/sft/workflow.py` 的 `run_sft()` 是最常用的训练路径：

```
1. load_tokenizer(model_args)
2. get_template_and_fix_tokenizer(tokenizer, data_args)
3. get_dataset(..., stage="sft")
4. load_model(tokenizer, model_args, finetuning_args)
5. SFTDataCollatorWith4DAttentionMask(...)
6. CustomSeq2SeqTrainer(...).train()
7. save_model / save_metrics / plot_loss
```

### 关键步骤

```python
def run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks):
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", ...)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = SFTDataCollatorWith4DAttentionMask(template=template, model=model, ...)

    trainer = CustomSeq2SeqTrainer(
        model=model, args=training_args, data_collator=data_collator, callbacks=callbacks, ...
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_metrics("train", train_result.metrics)
        if finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss"])
```

训练完成后输出：

| 产物 | 路径 | 说明 |
|------|------|------|
| 模型权重 | `output_dir/` | checkpoint 或 LoRA adapter |
| 训练指标 | `output_dir/train_results.json` | loss、learning_rate 等 |
| Trainer 状态 | `output_dir/trainer_state.json` | 断点续训用 |
| Loss 曲线 | `output_dir/training_loss.png` | `plot_loss=True` 时生成 |

## 微调方法

| 方法 | `finetuning_type` | 实现 | 说明 |
|------|-------------------|------|------|
| 全量微调 | `full` | `adapter._setup_full_tuning()` | 更新所有参数 |
| 部分层微调 | `freeze` | `adapter._setup_freeze_tuning()` | 仅训练指定层 |
| LoRA | `lora` | PEFT `LoraConfig` | 低秩适配，最常用 |
| OFT | `oft` | PEFT `OFTConfig` | 正交微调 |

LoRA 关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_rank` | 8 | 低秩矩阵的秩 |
| `lora_alpha` | 自动 | 缩放系数，通常 = 2 × rank |
| `lora_target` | all | 目标模块 |
| `lora_dropout` | 0 | Dropout 比例 |
| `use_rslora` | false | 是否使用 rsLoRA |
| `use_dora` | false | 是否使用 DoRA |

## 回调机制

`train/callbacks.py` 提供训练过程中的钩子：

| 回调 | 触发条件 | 功能 |
|------|---------|------|
| `LogCallback` | 始终 | 格式化训练日志 |
| `PissaConvertCallback` | `pissa_convert=True` | PiSSA → 标准 LoRA 转换 |
| `SwanLabCallback` | `use_swanlab=True` | SwanLab 实验追踪 |
| `EarlyStoppingCallback` | `early_stopping_steps` | 验证 loss 不下降时早停 |
| `TorchProfilerCallback` | `enable_torch_profiler` | PyTorch Profiler |
| `ModuleProfilerCallback` | `profile_modules` | 模块级性能分析 |
| `ReporterCallback` | 始终 | 训练结束报告 |

## 模型导出

`export_model()` 在 `tuner.py` 中实现，用于合并 LoRA 并导出完整权重：

```bash
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

流程：

```
1. get_infer_args() 解析参数
2. load_tokenizer() + get_template_and_fix_tokenizer()
3. load_model() 加载基座 + LoRA 适配器
4. 合并 LoRA 权重到基座模型
5. 按指定 dtype 保存到 export_dir
```

注意：量化模型不能直接合并 LoRA，需先合并再量化。

## 特殊训练后端

| 后端 | 开关 | 模块 | 支持 stage |
|------|------|------|-----------|
| HyperParallel FSDP2 | `use_hyper_parallel=True` | `train/hyper_parallel/` | pt, sft |
| Megatron-core | `use_mca=True` + `USE_MCA=1` | `train/mca/` | pt, sft, dpo |
| Ray | `use_ray=True` | `train/trainer_utils.py` | 全部 |
| KTransformers | `use_kt=True` | 模型加载层 | 大 MoE 模型 |

## 对齐训练简述

### DPO（Direct Preference Optimization）

- 数据格式：成对偏好（chosen / rejected）
- 无需单独训练 RM
- 支持多种损失：`pref_loss` = sigmoid / hinge / ipo / orpo / simpo
- 配置示例：`examples/train_lora/qwen3_lora_dpo.yaml`

### PPO（Proximal Policy Optimization）

- 需要预训练的 Reward Model
- 基于 TRL 实现策略梯度优化
- 三阶段：SFT → RM → PPO

### KTO（Kahneman-Tversky Optimization）

- 只需二元反馈（好/坏），不需要成对偏好
- 配置示例：`examples/train_lora/qwen3_lora_kto.yaml`

### RM（Reward Modeling）

- 训练 value head 为 PPO 提供奖励信号
- 配置示例：`examples/train_lora/qwen3_lora_reward.yaml`
