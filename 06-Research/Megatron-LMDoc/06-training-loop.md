# 06 - 训练循环

源码：`megatron/training/training.py`（4500+ 行）

## 调用链

```
pretrain_gpt.py __main__
  → parse_and_validate_args()
  → pretrain(cfg, dataset_provider, model_provider, forward_step, ...)
      → initialize_megatron()
      → setup_model_and_optimizer()
      → build train/valid/test iterators
      → train()  # 主循环
          → train_step()  # 每 iteration
              → forward_backward_func()
              → optimizer.step()
```

## pretrain() 参数（概念）

| 参数 | 类型 | 作用 |
|------|------|------|
| `cfg_container` | PretrainConfigContainer | 训练/checkpoint/validation 配置 |
| `train_valid_test_dataset_provider` | Callable | 构建数据集 |
| `model_provider` | Callable | 构建 GPTModel（可能多 chunk） |
| `forward_step_func` | Callable | 单 microbatch 前向+loss |
| `model_type` | ModelType | encoder/decoder 等 |

## train() 主循环

每 iteration：

1. 可选 RL rollout（`perform_rl_step`）
2. `train_step()` — forward-backward + optim
3. Logging（loss、lr、throughput、MFU）
4. 定期 `evaluate()` validation
5. `save_checkpoint_and_time()` 按 interval
6. 退出条件：达到 `train_iters`、duration、signal

## train_step()

```python
def train_step(forward_step_func, data_iterator, model, optimizer, ...):
    while rerun_state_machine.should_run_forward_backward(...):
        zero_grad(model)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            num_microbatches=get_num_microbatches(),
            ...
        )
    # optimizer.step(), scheduler, grad clip
    return loss_dict, skipped_iter, should_checkpoint, ...
```

**rerun_state_machine**：支持 deterministic rerun（调试/容错）。

## forward_backward_func

来自 pipeline parallel schedule：

- 无 PP：简单 forward-backward
- 有 PP：`get_forward_backward_func()` → 1F1B / interleaved

Schedule 负责 microbatch 在 stage 间的 P2P 与 bubble 优化。

## forward_step（GPT）

`pretrain_gpt.py` 中典型逻辑：

1. `batch = get_batch(data_iterator, vp_stage)`
2. `output_tensor = model(**batch_fields)`
3. `loss = loss_func(output_tensor, ...)`
4. return loss

SFT 模式：`SFTDataset` + packed `cu_seqlens`。

## 优化器

- `get_megatron_optimizer()`：Adam / AdamW + 可选 Muon
- **DistributedOptimizer**：optimizer state 分片
- **LayerWiseDistributedOptimizer**：按层分片
- LR scheduler：warmup + cosine/linear 等（args 控制）

## 梯度

- `finalize_model_grads()`：DP/TP 梯度同步收尾
- `clip_grad`：`--clip-grad` 全局范数裁剪
- `--calculate-per-token-loss`：token 级 loss 聚合

## Microbatch 与 Global Batch

`num_microbatches_calculator`：

- 根据 `global_batch_size`、`micro_batch_size`、DP size 计算
- 支持 **batch size rampup**（渐增 global batch）

## 容错与弹性

| 特性 | 说明 |
|------|------|
| `inprocess_restart` | 进程内重启 |
| `ft_integration` | Fault tolerance 集成 |
| `fault_injector` | 测试注入故障 |
| `elastification/` | Flextron 弹性 shape |

## RL 训练分支

`args.perform_rl_step` 时：

- 调用 `megatron.rl` rollout
- 可 `--skip-train` 仅推理收集
- 见 [08-inference-rl.md](./08-inference-rl.md)

## 性能监控

- `get_timers()`：细粒度 CUDA timer
- `StragglerDetector`：慢 rank 检测
- `one_logger`：NVIDIA 内部 metrics
- WandB：`wandb_utils.py`

## 关键训练参数（CLI 节选）

| 参数 | 含义 |
|------|------|
| `--train-iters` | 总 iteration |
| `--micro-batch-size` | 每 GPU 每 microbatch |
| `--global-batch-size` | 全局 batch |
| `--lr` / `--min-lr` | 学习率 |
| `--weight-decay` | 权重衰减 |
| `--clip-grad` | 梯度裁剪 |
| `--bf16` / `--fp16` | 混合精度 |
| `--use-distributed-optimizer` | 分片 optimizer |

完整列表见 `megatron/training/arguments.py`（极大）。

## 源码定位

| 函数 | 约略行号（training.py） |
|------|-------------------------|
| `pretrain()` | ~1004 |
| `train_step()` | ~2237 |
| `train()` | ~3199 |

（行号随版本漂移，以 grep 为准。）
