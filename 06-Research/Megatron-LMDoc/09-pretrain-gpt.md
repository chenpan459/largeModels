# 09 - pretrain_gpt.py 精读

文件：`/home/cp/work2/largeModels/06-Research/Megatron-LM/pretrain_gpt.py`

GPT **预训练 / SFT / FIM** 的标准入口。

## 文件结构

| 部分 | 内容 |
|------|------|
| imports | Core datasets、mpu、training API |
| `BATCH_KEYS` | batch 字典键序（与 unpack 一致） |
| `get_batch()` | 取 batch + TP/PP/CP 分发 |
| `forward_step()` | 前向与 loss |
| `loss_func()` | LM loss 计算 |
| `train_valid_test_datasets_provider()` | 构建数据集 |
| `get_embedding_ranks()` | PP embedding 组 rank |
| `__main__` | parse args → `pretrain()` |

## __main__ 流程

```python
if __name__ == "__main__":
    set_startup_timestamps(...)
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    args = parse_and_validate_args(...)
    model_cfg = gpt_config_from_args(args)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
```

## get_batch 逻辑要点

1. 非首末 PP stage 且无 MTP/packed seq 时返回 `[None] * len(BATCH_KEYS)`
2. **TP rank 0** 调用 `next(data_iterator)`
3. CP/TP broadcast batch 到同组 rank
4. 可选在 dataloader 构建 `attention_mask`
5. SFT / inter-document masking 使用 `cu_seqlens`

## 数据集选择

```python
if args.sft:
    dataset_type = SFTDataset
elif args.mock_data:
    dataset_type = MockGPTDataset
elif args.fim_data:
    dataset_type = GPTFIMDataset
else:
    dataset_type = GPTDataset
```

## model_provider

通过 `gpt_builders.py` + `training/models/gpt.py`：

- 构建 `GPTModel` + `TransformerConfig`
- 按 PP rank 设置 `pre_process` / `post_process`
- Virtual PP 多 chunk

## forward_step

典型步骤：

1. 从 iterator 取 batch（或接收 schedule 传入）
2. 调用 model forward
3. 应用 `loss_func`（token-level CE，含 MTP）
4. 返回 `(output_tensor, loss_func_partial)` 供 schedule 使用

## 扩展点

| 扩展 | 方式 |
|------|------|
| 新 loss | 自定义 `loss_func` |
| 新数据 | 新 Dataset 类 + provider |
| ModelOpt | `add_modelopt_args` + `loss_func_modelopt` |
| 新模型变体 | 修改 `gpt_builder` / layer spec |

## 相关文件

| 文件 | 关系 |
|------|------|
| `gpt_builders.py` | 模型构建器 |
| `model_provider.py` | model_provider 包装 |
| `megatron/training/arguments.py` | 全部 CLI |
| `megatron/training/models/gpt.py` | GPTModelBuilder |

## 最小运行示例（mock 数据）

```bash
torchrun --nproc_per_node=1 pretrain_gpt.py \
  --mock-data \
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --micro-batch-size 2 \
  --global-batch-size 2 \
  --train-iters 10 \
  --lr 1e-4 \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-size 50257 \
  --seq-length 512 \
  --max-position-embeddings 512
```

（参数名以当前版本 `arguments.py` 为准，不同版本可能略有差异。）

## 与其他 pretrain 脚本

| 脚本 | 差异 |
|------|------|
| `pretrain_mamba.py` | Mamba/hybrid layer spec |
| `pretrain_vlm.py` | 多模态 batch + vision tokenizer |
| `examples/bert/pretrain_bert.py` | BERT MLM |

核心 `pretrain()` 框架相同，换 model_provider + forward_step。
