# 06 - Model Executor 与 GPUModelRunner

## model_executor 概览

路径：`vllm/model_executor/`（~598 文件）

```
model_executor/
├── model_loader/       # 从 HF/Safetensors 加载权重
├── models/             # 各架构 forward 实现
│   ├── llama.py, qwen2.py, deepseek_v2.py
│   ├── mixtral.py, qwen3_moe.py
│   └── registry.py     # 架构名 → 类映射
├── layers/             # 可复用层
│   ├── linear.py, rotary_embedding.py
│   ├── fused_moe.py    # MoE 融合
│   ├── sampler.py
│   └── quantization/   # GPTQ/AWQ/FP8/Marlin...
└── parameter.py        # 权重参数封装
```

## 模型注册

`models/registry.py` — `_MODELS` 字典映射 HuggingFace `architectures` 到 Python 类。

添加新模型：实现 `nn.Module` + 注册 + 可选 `transformers_utils/configs/`。

## 权重加载

`model_loader/loader.py`：

- 支持 HF Hub、本地路径、S3
- `tensorizer` 快速加载
- 量化格式检测（`quant_config`）
- TP 分片加载（每 rank 只加载所属 shard）

```python
from vllm.model_executor.model_loader import get_model
model = get_model(vllm_config)
```

## GPUModelRunner

文件：`vllm/v1/worker/gpu_model_runner.py`（1700+ 行）

**单 GPU 上的执行单元**：

| 阶段 | 行为 |
|------|------|
| 初始化 | `get_model()`、分配 KV、选 attention backend |
| `execute_model` | 解析 `SchedulerOutput` → `InputBatch` |
| Forward | 设置 `ForwardContext`，跑 transformer |
| Sample | `Sampler` / `RejectionSampler`（spec decode） |
| 返回 | `ModelRunnerOutput` |

关键依赖：

- `FlashAttentionMetadata`（`v1/attention/backends/flash_attn.py`）
- `InputBatch` / `CachedRequestState`（`gpu_input_batch.py`）
- `EagleProposer`、`NgramProposer`（spec decode）

## ForwardContext

`forward_context.py` — 线程局部上下文：

- 当前 batch 的 metadata
- CUDA graph capture 模式
- 供 `Attention` 层读取 block table

## 量化层

`layers/quantization/` — 按方法分目录：

| 方法 | 目录/类 |
|------|---------|
| GPTQ | `gptq.py` |
| AWQ | `awq.py` |
| FP8 | `fp8.py` + utils |
| Marlin | `marlin.py` |
| Compressed_tensors | 新格式 |

量化与 TP 结合：每 rank 持有对应 shard 的量化权重。

## MoE 执行

`layers/fused_moe.py` — `FusedMoE`：

- TopK router
- 与 EP/TP 配合的 token dispatch
- 融合 grouped GEMM（Cutlass/TE）

模型：`mixtral.py`、`qwen3_moe.py`、`deepseek_v2.py` 等。

## LoRA

`v1/worker/lora_model_runner_mixin.py` — 动态 LoRA adapter：

- 运行时 load/unload
- OpenAI API `/v1/load_lora_adapter`

## Pooling / Embedding 模式

`runner_type != "generate"` 时走：

- `pooling_model_runner.py` — embedding/rerank/classify
- API：`/v1/embeddings`、`/v1/rerank`

## CPU / TPU / 其他

| Runner | 路径 |
|--------|------|
| `cpu_model_runner.py` | CPU 推理 |
| `tpu_model_runner.py` | Google TPU |
| `neuron_model_runner.py` | AWS Neuron |

V1 主要维护 `gpu_model_runner.py`。

## torch.compile

`config.compilation_config` — `CompilationLevel` 控制 `torch.compile` 范围，与 V1 执行循环集成。

## 阅读顺序

1. `models/llama.py` — 典型 decoder
2. `attention/layer.py` — KV + backend 调用
3. `gpu_model_runner.py` — `execute_model` 主路径
4. 目标架构（如 `deepseek_v2.py`）— MLA/MoE

## 与训练框架

| 训练 | 推理（vLLM） |
|------|--------------|
| Megatron 分片 checkpoint | Bridge → HF → vLLM loader |
| LLaMA-Factory adapter | LoRA mixin |

权重格式通常为 **HuggingFace**，非 GGUF（GGUF 用 llama.cpp）。
