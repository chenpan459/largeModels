# 15 - 量化方案目录

路径：`vllm/model_executor/layers/quantization/`（72 文件）

## 概述

vLLM 量化在 **层级别** 应用：每个 `Linear`/`MoE` 层根据 `QuantizationConfig` 选择对应的 `QuantizeMethodBase` 实现。加载权重时 `configure_quant_config()` 将 HF 量化 metadata 映射到 vLLM 内核。

```bash
vllm serve model --quantization awq
# 或模型自带 quantization_config（HF config.json）
```

## 架构

```
QuantizationConfig (base_config.py)
  → get_quant_method(layer, prefix)
  → QuantizeMethodBase
      → create_weights()     # 分配量化权重 buffer
      → apply()                # forward 时 dequant + matmul
      → process_weights_after_loading()  # Marlin repack 等
```

Schema 验证：`quantization/schema.py`

## 主要量化方案

| 方法 | 文件 | 内核 | 说明 |
|------|------|------|------|
| **GPTQ** | `gptq.py` | CUDA GPTQ | 4bit 权重量化 |
| **GPTQ Marlin** | `gptq_marlin.py` | Marlin | GPTQ + Marlin 加速 |
| **AWQ** | `awq.py` | CUDA AWQ | 4bit activation-aware |
| **AWQ Marlin** | `awq_marlin.py` | Marlin | AWQ + Marlin |
| **AWQ Triton** | `awq_triton.py` | Triton | ROCm/回退 |
| **FP8** | `fp8.py` | Cutlass/TE | H100+ FP8 |
| **FBGEMM FP8** | `fbgemm_fp8.py` | FBGEMM | Meta FP8 |
| **PTPC FP8** | `ptpc_fp8.py` | PTPC | Per-token per-channel |
| **Marlin** | `marlin.py` | Marlin | 通用 Marlin int4 |
| **GGUF** | `gguf.py` | ggml 兼容 | **V1 不支持** |
| **BitsAndBytes** | `bitsandbytes.py` | BnB | 8bit/4bit NF4 |
| **AQLM** | `aqlm.py` | AQLM | Additive quant |
| **Quark** | `quark.py` | AMD Quark | |
| **TorchAO** | `torchao.py` | torchao | PyTorch 原生 |
| **ModelOpt** | `modelopt.py` | NVIDIA ModelOpt | |
| **DeepSpeed FP** | `deepspeedfp.py` | DS | |
| **HQQ** | `hqq_marlin.py` | HQQ | Half-Quadratic |
| **QQQ** | `qqq.py` | QQQ | |
| **NVFP4** | `nvfp4.py` | NVFP4 | Blackwell |
| **Experts Int8** | `experts_int8.py` | MoE int8 | |
| **IPEX AWQ** | `ipex_awq.py` | Intel IPEX | CPU/XPU |
| **TPU Int8** | `tpu_int8.py` | TPU | |

## Compressed Tensors

目录：`quantization/compressed_tensors/` — Neural Magic 格式

| 子模块 | 说明 |
|--------|------|
| `compressed_tensors.py` | 主入口 |
| `compressed_tensors_moe.py` | MoE 支持 |
| `triton_scaled_mm.py` | Triton scaled matmul |
| `schemes/` | 多种压缩 scheme |

支持多种 scheme（W4A16、W8A8 等），通过 config 自动选择 kernel。

## 内核子目录

| 目录 | 用途 |
|------|------|
| `kernels/` | 通用量化 kernel |
| `kernels/mixed_precision/` | 混合精度 GEMM |
| `kernels/scaled_mm/` | scaled matrix multiply |
| `utils/` | 量化工具函数 |

## 与 TP 的结合

量化权重按 TP 分片加载：

```python
# loader.py
tp_rank = get_tensor_model_parallel_rank()
# 每 rank 只加载属于它的 shard
```

Marlin repack 在 `process_weights_after_loading()` 中 **per-rank** 执行。

MoE 量化：`fused_moe.py` + `experts_int8.py` / `compressed_tensors_moe.py`

## FP8 详解

| 变体 | 场景 |
|------|------|
| `fp8.py` | 标准 FP8 weight + activation |
| `fbgemm_fp8.py` | FBGEMM backend |
| `ptpc_fp8.py` | Per-token per-channel 缩放 |
| FP8 KV cache | `cache_dtype=fp8`（与 weight FP8 独立） |

FP8 权重量化通常需 **H100+**（SM 89+）或支持的 AMD GPU。

FP8 KV cache：

- V0：需 FlashInfer attention
- V1：需 FlashAttention FP8 支持

## 如何选择量化方案

| 场景 | 推荐 |
|------|------|
| 生产 NVIDIA 4bit | AWQ Marlin 或 GPTQ Marlin |
| H100 吞吐 | FP8 |
| 已有 AWQ 权重 | `--quantization awq` |
| 已有 GPTQ 权重 | `--quantization gptq` |
| Compressed HF 模型 | auto 检测 compressed-tensors |
| CPU 推理 | `--quantization bitsandbytes` 或 IPEX |
| GGUF | 用 llama.cpp，非 vLLM V1 |

## 加载流程中的量化

```
1. HF config.json → quantization_config 字段
2. get_quant_config() 解析为 QuantizationConfig 子类
3. loader 迭代 state_dict
4. 每层 Linear 调用 quant_method.create_weights()
5. 加载量化权重 tensor
6. process_weights_after_loading() → Marlin repack / scale 校验
7. warmup forward 验证 kernel 可用
```

## 添加新量化方法

1. 继承 `QuantizationConfig` + `QuantizeMethodBase`
2. 实现 `get_quant_method()`、`create_weights()`、`apply()`
3. 在 `quantization/__init__.py` 注册方法名
4. 添加对应 kernel（Triton/CUDA）
5. 测试 TP 兼容性

## 性能提示

| 提示 | 说明 |
|------|------|
| Marlin > 原生 GPTQ/AWQ | 生产首选 Marlin 变体 |
| 预热 | 首次 forward 可能触发 JIT/repack |
| `--enforce-eager` | 量化 + graph 可能冲突，调试时用 |
| TP + 量化 | 确保每 rank shard 完整 |

## 与 llama.cpp 量化对照

| vLLM | llama.cpp |
|------|-----------|
| AWQ/GPTQ Marlin | Q4_K_M 等 GGUF quants |
| FP8 (HF) | 部分 GGUF F16/BF16 |
| 运行时加载 HF quant | 预量化 GGUF 文件 |
| GPU kernel 多样 | ggml quants |

GGUF 量化和加载见 llama.cpp；vLLM 主路径是 **HuggingFace 量化格式**。

## 关键文件

| 文件 | 内容 |
|------|------|
| `base_config.py` | QuantizeMethodBase 抽象 |
| `schema.py` | 配置 schema |
| `__init__.py` | 方法名 → 类注册 |
| `gptq_marlin.py` | 常用 Marlin GPTQ |
| `awq_marlin.py` | 常用 Marlin AWQ |
| `fp8.py` | FP8 主实现 |
