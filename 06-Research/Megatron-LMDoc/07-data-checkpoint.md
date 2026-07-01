# 07 - 数据与 Checkpoint

## 数据管线总览

```mermaid
flowchart LR
    RAW[原始文本/jsonl] --> PRE[tools/preprocess_data.py]
    PRE --> BIN[.bin + .idx]
    BIN --> DS[GPTDataset / SFTDataset]
    DS --> BLEND[BlendedMegatronDatasetBuilder]
    BLEND --> IT[data_iterator]
    IT --> GB[get_batch in pretrain_gpt.py]
    GB --> MODEL[GPTModel forward]
```

## 预处理

```bash
cd /home/cp/work2/largeModels/06-Research/Megatron-LM

python tools/preprocess_data.py \
  --input /path/to/corpus.jsonl \
  --output-prefix /path/to/my_corpus \
  --tokenizer-type GPT2BPETokenizer \
  --workers 8
```

产出：

| 文件 | 内容 |
|------|------|
| `my_corpus.bin` | token id 连续二进制 |
| `my_corpus.idx` | 文档边界索引（可 seek） |

多语料合并：`tools/merge_datasets.py`

多模态：`tools/preprocess_mmdata.py`

## 数据集类

| 类 | 路径 | 用途 |
|----|------|------|
| `GPTDataset` | `core/datasets/gpt_dataset.py` | 标准 causal LM 预训练 |
| `MockGPTDataset` | 同上 | 随机数据、CI/调试 |
| `GPTFIMDataset` | `training/datasets/fim_dataset.py` | Fill-in-Middle |
| `SFTDataset` | `training/datasets/sft_dataset.py` | 监督微调 |

### BlendedMegatronDatasetBuilder

`pretrain_gpt.py` 中构建 train/valid/test：

```python
train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
    dataset_type, train_val_test_num_samples, is_dataset_built, config
).build()
```

- 多 corpus 按 **权重 blend**（data config 或 CLI）
- `is_dataset_built_on_rank` 控制哪些 PP rank 参与构建

### 数据集选择逻辑（pretrain_gpt.py）

```python
if args.sft:
    dataset_type = SFTDataset          # packed sequence
elif args.mock_data:
    dataset_type = MockGPTDataset
elif args.fim_data:
    dataset_type = GPTFIMDataset
else:
    dataset_type = GPTDataset
```

## get_batch 与并行切分

`pretrain_gpt.py` — `BATCH_KEYS` 标准字段：

- `tokens`, `labels`, `loss_mask`, `position_ids`
- `attention_mask`（可选 dataloader 内构建）
- `cu_seqlens`, `max_seqlen`（SFT / packed / inter-document masking）
- CP：`hybrid_cp_group`, `local_cp_size`

流程要点：

1. 非首末 PP stage 且无 MTP/packed 时返回空 batch
2. **TP rank 0** 从 iterator 取 batch，再 broadcast 到 TP group
3. SFT 使用 `PackedSeqParams`（`qkv_format="thd"`）喂 TE varlen attention

`forward_step` 中：

```python
packed_seq_params = PackedSeqParams(
    qkv_format="thd",
    cu_seqlens_q=cu_seqlens_for_params,
    max_seqlen_q=int(max_seqlen.item()),
    ...
)
output = model(tokens, position_ids, attention_mask, labels=labels,
               packed_seq_params=packed_seq_params)
```

## Tokenizer

- `megatron/core/tokenizers/` + `build_tokenizer()`
- 类型：GPT2 BPE、SentencePiece、HuggingFace 等
- `vocab_utils.calculate_padded_vocab_size()` — TP 对齐 vocab 大小

## Checkpoint 体系

### Legacy vs Distributed

| 类型 | 说明 | 推荐 |
|------|------|------|
| Legacy | 每 rank 一个 `.pt` | 旧项目迁移 |
| **Dist checkpoint** | PyTorch distributed checkpoint + 分片 metadata | **生产默认** |

启用：`--use-dist-ckpt`

### training/checkpointing.py

主要 API：

- `save_checkpoint(iteration, model, optimizer, ...)`
- `load_checkpoint(model, optimizer, ...)`
- `schedule_async_save` — 异步 IO 重叠训练

处理内容：

- TP/PP/EP 分片 state dict（`ShardedStateDict`）
- `DistributedOptimizer` 分片
- FP8 extra state、MoE expert、SwiGLU shard
- RNG、iteration、args 元数据
- FSDP / uneven dtensor 变体

### dist_checkpointing/

路径：`megatron/core/dist_checkpointing/`

- `ShardedTensor` mapping：logical name → global tensor + offsets
- `FullyParallelSaveStrategyWrapper` — 并行写盘
- 支持 S3 等远程存储（`MultiStorageClientFeature`）

### 转换工具

`tools/checkpoint/`：

| 工具 | 作用 |
|------|------|
| `convert.py` | 通用转换入口 |
| `loader_core.py` / `saver_core.py` | Megatron Core 格式 |
| `loader_llava.py` | 多模态 |
| `hybrid_conversion.py` | Hybrid 模型 |
| `checkpoint_inspector.py` | 检查分片结构 |

**HF 互转**推荐 [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)，避免手工 reshape TP 分片。

## 恢复与微调

```bash
torchrun ... pretrain_gpt.py \
  --load /path/to/checkpoint \
  --save /path/to/checkpoint \
  --save-interval 1000 \
  --no-load-optim          # 仅权重
  --finetune               # 允许结构部分不匹配
```

## 理论显存估算

- `tools/report_theoretical_memory.py`
- `training/theoretical_memory_usage.py`

用于确定 TP/PP 度与 microbatch 上限。

## 与本仓库其他模块

| 场景 | 路径 |
|------|------|
| 小规模 RAG 文档 | `07-业务应用/kefu-kb/data/docs/` |
| HF 微调数据 | `02-训练/LLaMA-Factory` |
| 推理部署 | Bridge → HF → `03-推理部署/vllm` |

Megatron 面向 **TB 级预训练 corpus**；业务 FAQ 通常不需 Megatron 数据管线。

## 实践 checklist

1. `preprocess_data.py` → `.bin/.idx`
2. CLI 配置 `--data-path`、`--split`、`--seq-length`
3. 首次训练 `--save` 目录为空
4. 多机共享文件系统或 object store 存 checkpoint
5. 部署前 Bridge 转 HF，勿直接喂 vLLM 分片 ckpt
