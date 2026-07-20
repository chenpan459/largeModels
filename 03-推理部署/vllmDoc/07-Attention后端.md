# 07 - Attention Backends

## 抽象层架构

vLLM 将 Attention 分为三层：

```
Attention (layer.py)           # nn.Module，QKV proj + 调用 backend
    ↓
AttentionBackend (abstract.py) # 工厂：impl、metadata、builder
    ↓
AttentionImpl                  # 实际 kernel（FlashAttn、Triton、MLA...）
```

### abstract.py 核心类型

文件：`vllm/attention/backends/abstract.py`

| 类型 | 职责 |
|------|------|
| `AttentionType` | `DECODER`、`ENCODER`、`ENCODER_ONLY`、`ENCODER_DECODER` |
| `AttentionBackend` | 工厂：`get_impl_cls`、`get_metadata_cls`、`get_builder_cls`、`get_kv_cache_shape`、`swap_blocks`、`copy_blocks` |
| `AttentionMetadata` | batch 级 metadata：`num_prefills`、`slot_mapping`、`seq_lens` 等 |
| `AttentionImpl` | `forward(query, key, value, kv_cache, attn_metadata)` |
| `AttentionMetadataBuilder` | 从 ModelRunner 状态构建 metadata |
| `AttentionState` | backend 持久状态 |

`AttentionMetadata` 关键字段：

```python
num_prefills: int
num_prefill_tokens: int
num_decode_tokens: int
slot_mapping: Tensor       # token → physical slot
# V1 还有 query_start_loc、block_table 等
```

### Attention 模块

`attention/layer.py`：

```python
class Attention(nn.Module):
    """1. Store K/V in KV cache
       2. Perform MHA/GQA/MQA/MLA
       3. Return output"""
```

Forward 流程：

1. Q/K/V linear projection
2. 可选 QK norm（部分模型）
3. 调用 `attn_backend.get_impl_cls()(self, ...)` 执行 attention
4. Output projection

## Backend 选择

入口：`attention/selector.py` → `get_attn_backend()`

```python
def get_attn_backend(head_size, dtype, kv_cache_dtype, block_size, ...):
    return current_platform.get_attn_backend_cls(...)
```

平台实现：

- `platforms/cuda.py` — NVIDIA（主力）
- `platforms/rocm.py` — AMD
- `platforms/tpu.py` — Google TPU

环境变量覆盖：`VLLM_ATTENTION_BACKEND` → `_Backend` enum

### _Backend 枚举（`platforms/interface.py`）

```
FLASH_ATTN, FLASH_ATTN_VLLM_V1, TRITON_ATTN_VLLM_V1,
XFORMERS, FLASHINFER, TRITON_MLA, FLASHMLA,
PALLAS, PALLAS_VLLM_V1, ...
```

## V1 Backends

路径：`vllm/v1/attention/backends/`

| Backend | 文件 | Backend 名 | 场景 |
|---------|------|------------|------|
| FlashAttention V1 | `flash_attn.py` | `FLASH_ATTN_VLLM_V1` | NVIDIA 默认（SM≥80） |
| Triton V1 | `triton_attn.py` | `TRITON_ATTN_VLLM_V1` | 通用/回退 |
| MLA Flash | `mla/flashmla.py` | `FLASHMLA` | DeepSeek MLA（block=64） |
| MLA Triton | `mla/triton_mla.py` | `TRITON_MLA` | MLA 回退 |
| Pallas | `pallas.py` | TPU | Google TPU |

### V1 CUDA 选择逻辑（`platforms/cuda.py:215-293`）

**V1 路径（`use_v1=True`）**：

```
1. use_mla=True?
   → FlashMLA（若支持，block_size=64）或 Triton MLA
2. 显式 TRITON_ATTN_VLLM_V1 → Triton V1
3. SM ≥ 80 → FlashAttentionBackend V1（默认）
4. 否则 → 报错或 Triton
```

**V0 路径**：

```
FlashInfer / XFormers / FlashAttn
FP8 KV → 需 FlashInfer（否则 warning + fallback）
```

### V1 FlashAttention 细节

`flash_attn.py`：

- `FlashAttentionMetadata`：`query_start_loc`、`seq_lens`、block table
- `FlashAttentionMetadataBuilder`：从 `GPUModelRunner` weakref 构建
- KV shape：`(2, num_blocks, block_size, num_kv_heads, head_size)`
- 支持 cascade attention（`num_common_prefix_blocks`）

## V0 Backends

路径：`vllm/attention/backends/`

| Backend | 文件 | 说明 |
|---------|------|------|
| FlashAttention | `flash_attn.py` | 经典 paged FA |
| FlashInfer | `flashinfer.py` | 高性能 decode；**V1 attention 不可用** |
| XFormers | `xformers.py` | 回退 |
| ROCm Flash | `rocm_flash_attn.py` | AMD |
| MLA | `mla/` | DeepSeek V2/V3 |

**重要**：FlashInfer 在 V1 中仅用于 **采样**（top-k/top-p），不用于 attention。V1 oracle 拒绝 `VLLM_ATTENTION_BACKEND=FLASHINFER`。

## FlashInfer 的两个用途

| 用途 | 路径 | V1 |
|------|------|-----|
| Attention | `attention/backends/flashinfer.py` | ✗ |
| Sampling top-k/top-p | `v1/sample/ops/topk_topp_sampler.py` | ✓（`VLLM_USE_FLASHINFER_SAMPLER`） |

不要混淆两者。

## GQA / MQA

`num_kv_heads < num_heads` 时：

- Backend 内处理 KV head 广播或压缩布局
- 与 HuggingFace / Megatron 数学一致
- `num_kv_heads=1` 即 MQA

## MLA（Multi-Latent Attention）

DeepSeek-V2/V3 等：

- `use_mla=True` → `mla/` backend
- 低秩 KV 压缩 + absorbed 版本（减少 decode 带宽）
- FlashMLA 要求 `block_size=64`（platform 自动调整）
- 环境变量 `VLLM_MLA_DISABLE=True` 强制禁用

文件：

- `v1/attention/backends/mla/common.py` — 共享逻辑
- `mla/flashmla.py` — 高性能 kernel
- `mla/triton_mla.py` — 回退

## Sliding Window

`SlidingWindowSpec`（`kv_cache_interface.py`）：

- 部分层仅保留 window 内 KV
- Block 管理仍用 paged 框架
- `specialized_manager.py` 处理 window 外 block 回收

## Cascade Attention

当 batch 内多 request 共享前缀 block 时：

- Scheduler 输出 `num_common_prefix_blocks`
- Backend 可合并公共前缀的 attention 计算
- 与 prefix caching 协同

## Custom Ops 与 CUDA Graph

```python
direct_register_custom_op(...)  # 注册 PyTorch custom op
```

目的：

- Piecewise CUDA graph 捕获时 op 边界清晰
- `accept_output_buffer=True` 时在 graph 内预分配 output

## FP8 KV Cache 与 Backend

| 场景 | 要求 |
|------|------|
| V0 + fp8 KV | FlashInfer attention |
| V1 + fp8 KV | FlashAttention FP8 支持（oracle 检查） |

## 性能要点

| 阶段 | 瓶颈 | 优化 |
|------|------|------|
| Prefill | Compute-bound | FA varlen、大 matmul |
| Decode | Memory-bound | GQA/MLA 减 KV 读、CUDA graph |
| 小 batch decode | Kernel launch | CUDA graph capture |
| 长上下文 | KV 容量 | PagedAttention + prefix cache |

调试：`--enforce-eager` 禁用 graph；日志 `Using attention backend: ...`

## V0 vs V1 Backend 对照

| | V0 | V1 |
|---|----|----|
| 路径 | `attention/backends/` | `v1/attention/backends/` |
| Metadata | `AttentionMetadata` 子类 | `FlashAttentionMetadata` + Builder |
| 默认 NVIDIA | FlashAttn / FlashInfer | FlashAttn V1 |
| swap_blocks | 支持（CPU swap） | 不使用 |
| Builder 模式 | V0 runner 内联 | `AttentionMetadataBuilder` |

## 与 llama.cpp 对照

| vLLM | llama.cpp |
|------|-----------|
| FlashAttention paged | ggml flash attn |
| Backend selector | `--flash-attn` |
| MLA backend | DeepSeek 专用 kernel |
| Block table | KV cell index |

## 调试清单

| 现象 | 排查 |
|------|------|
| OOM | ↓ batch/seq，查 KV block 数 |
| 输出乱码 | dtype、RoPE base、GQA head 数 |
| 慢 decode | 确认 FA V1 非 Triton fallback |
| FP8 报错 | 检查 FA FP8 支持或换 backend |
| MLA 报错 | block_size=64、FlashMLA 可用性 |

## 关键源码行号

| 主题 | 位置 |
|------|------|
| 抽象类 | `attention/backends/abstract.py` |
| 选择器 | `attention/selector.py` |
| CUDA V1 选择 | `platforms/cuda.py:215-293` |
| V1 FA metadata | `v1/attention/backends/flash_attn.py:71+` |
