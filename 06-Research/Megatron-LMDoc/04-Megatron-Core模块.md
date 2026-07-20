# 04 - Megatron Core 模块

路径：`megatron/core/`（约 555 个 Python 文件）

## 顶层模块地图

```
megatron/core/
├── parallel_state.py      # MPU / process groups
├── model_parallel_config.py
├── process_groups_config.py # ProcessGroupCollection（新 API）
├── models/                  # 完整模型
│   ├── gpt/                 # GPTModel、layer specs
│   ├── T5/ BERT/ vision/ mamba/ ...
├── transformer/             # 构建块
│   ├── attention.py
│   ├── mlp.py
│   ├── transformer_layer.py
│   ├── transformer_block.py
│   ├── moe/                 # MoE 全套
│   ├── multi_latent_attention.py
│   └── multi_token_prediction.py
├── tensor_parallel/
├── pipeline_parallel/
├── distributed/             # DDP、FSDP
├── optimizer/               # 分布式优化器、Muon
├── datasets/                # GPTDataset 等
├── dist_checkpointing/      # 分片 checkpoint 策略
├── inference/               # 推理 context、engines
├── export/                  # TRT-LLM 等
├── fusions/                 # 融合 op
├── fp8_utils.py
└── enums.py
```

## GPTModel

文件：`megatron/core/models/gpt/gpt_model.py`

```python
class GPTModel(LanguageModule):
    """GPT Transformer language model."""
```

### 主要组件

| 组件 | 说明 |
|------|------|
| `LanguageModelEmbedding` | token + optional position embedding |
| `RotaryEmbedding` / `YarnRotaryEmbedding` | RoPE / YaRN 扩展 |
| `TransformerBlock` | N 层 decoder |
| `ColumnParallelLinear` | output logits（vocab 并行） |
| `MultiTokenPredictionBlock` | 可选 MTP 头 |

### Pipeline 相关参数

- `pre_process=True`：含 embedding（PP 首 stage）
- `post_process=True`：含 output head（PP 末 stage）
- 中间 stage 仅 transformer layers

### 推理模式

- `BaseInferenceContext`：KV cache 管理
- `InferenceCudaGraphScope`：CUDA Graph 捕获范围
- `paged_stash`：MoE 推理分页 stash

## TransformerBlock / Layer

| 类 | 文件 | 职责 |
|----|------|------|
| `TransformerBlock` | `transformer_block.py` | 堆叠 layers + 重计算策略 |
| `TransformerLayer` | `transformer_layer.py` | Self-Attn + MLP/MoE + LN |
| `DotProductAttention` | `dot_product_attention.py` | 核心 attention 计算 |
| `MLP` | `mlp.py` | SwiGLU/GeGLU 等 |

`TransformerConfig` 控制：head 数、GQA groups、qk layernorm、recompute 粒度等。

## Layer Specs

路径：`megatron/core/models/gpt/gpt_layer_specs.py`

通过 `ModuleSpec` 声明子模块工厂，例如：

- `get_gpt_layer_with_transformer_engine_spec` — TE 融合
- `get_gpt_layer_local_spec` — 纯 PyTorch
- `get_gpt_layer_with_inference_spec` — 推理优化

`training/models/gpt.py` 的 `GPTModelBuilder` 根据 args 选择 spec。

## Transformer Engine 集成

- 可选依赖 `transformer_engine`
- FP8 recipe、融合 attention、GroupedGEMM
- `transformer_impl` 配置项：`local` | `transformer_engine` | `inference_optimized`

## 分布式包装

| 包装 | 路径 | 用途 |
|------|------|------|
| `DistributedDataParallel` | `distributed/` | 经典 DDP + grad bucket |
| `FullyShardedDataParallel` | `distributed/fsdp/` | FSDP2 风格 |
| `DistributedOptimizer` | `optimizer/distrib_optimizer.py` | 分片 optimizer state |

## 数据集 Core

路径：`megatron/core/datasets/`

| 类 | 说明 |
|----|------|
| `GPTDataset` | Indexed .bin + .idx 格式 |
| `MockGPTDataset` | 随机数据测试 |
| `BlendedMegatronDatasetBuilder` | 多 corpus 按权重混合 |
| `MegatronDataset` | 基类 |

配合 `tools/preprocess_data.py` 预处理原始文本。

## Dist Checkpointing

路径：`megatron/core/dist_checkpointing/`

- **ShardedStateDict**：按 TP/PP/EP rank 描述 tensor 分片
- **Save/Load Strategy**：torch distributed checkpoint、fully parallel wrapper
- **Async save**：`schedule_async_save` 重叠 IO

与 `training/checkpointing.py` 集成。

## Inference Core

路径：`megatron/core/inference/`

- Text generation server（`tools/run_text_generation_server.py` 使用）
- Static / dynamic batching
- MoE inference grouped GEMM backend

## 其他 Core 子系统

| 模块 | 说明 |
|------|------|
| `rerun_state_machine` | 确定性重跑 / 故障恢复 |
| `full_cuda_graph` | 全图 CUDA Graph |
| `fault_injector` | 测试容错 |
| `quantization/` | FP8/FP4 recipe |
| `tokenizers/` | 多模态 tokenizer |

## mpu 别名

```python
from megatron.core import mpu  # 即 parallel_state 的便捷入口
```

训练代码中常见：`mpu.get_tensor_model_parallel_rank()`。

## 阅读顺序

1. `transformer_config.py` — 理解可配项
2. `gpt_model.py` — 模型组装
3. `transformer_layer.py` + `attention.py`
4. `tensor_parallel/layers.py`
5. `pipeline_parallel/schedules.py`
6. `transformer/moe/` — 若做 MoE

## 与 HuggingFace 差异

| HF | Megatron Core |
|----|---------------|
| 单进程单模型 | 显式并行切分 |
| `from_pretrained` | dist checkpoint + Bridge 转换 |
| `AutoModel` | `GPTModel` + ModuleSpec + TE |

转换见 **Megatron Bridge** 或 `tools/checkpoint/convert.py`。
