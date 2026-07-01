# 07 - Attention Backends

## 抽象层

`vllm/attention/layer.py` — 统一 `Attention` 模块：

```python
class Attention(nn.Module):
    """1. Store K/V in KV cache
       2. Perform MHA/GQA/MQA/MLA
       3. Return output"""
```

Backend 通过 `get_attn_backend()`（`attention/selector.py`）按 **硬件、dtype、head 布局、MLA** 选择。

## V1 Backends

路径：`vllm/v1/attention/backends/`

| Backend | 文件 | 场景 |
|---------|------|------|
| FlashAttention | `flash_attn.py` | NVIDIA 主力 |
| Triton | `triton_attn.py` | 通用/回退 |
| Pallas | `pallas.py` | TPU |
| MLA Flash | `mla/flashmla.py` | DeepSeek MLA |
| MLA Triton | `mla/triton_mla.py` | MLA 回退 |

### FlashAttentionMetadata

`flash_attn.py` 构建 paged attention 所需 metadata：

- `query_start_loc`、`seq_lens`
- block table 指针
- prefill/decode 分支（varlen）

与 **PagedAttention** 紧耦合：kernel 按 block 读 K/V。

## V0 Backends

`vllm/attention/backends/` — 历史实现：

- `flash_attn.py`、`xformers.py`、`rocm_flash_attn.py`
- `mla/` 子目录

`VLLM_USE_V1=1` 时 V1 runner 优先用 `v1/attention/backends/`。

## Backend 选择逻辑

`current_platform`（`platforms/cuda.py` 等）考虑：

- GPU 型号（A100/H100/B200）
- CUDA 版本
- dtype（FP16/BF16/FP8）
- 是否 MLA / sliding window
- env 覆盖（如 `VLLM_ATTENTION_BACKEND`）

## GQA / MQA

`num_kv_heads < num_heads` 时在 backend 内处理 KV 重复或压缩布局，与 Megatron/llama 一致。

## MLA（Multi-Latent Attention）

DeepSeek-V2/V3：

- `use_mla=True` 时走 `mla/` backend
- 低秩 KV + 吸收版（`absorbed_mla` 实验路径）
- 显著降低 decode 带宽

## Sliding Window

`SlidingWindowSpec` — 部分层仅保留 window 内 KV，block 管理仍用 paged 框架。

## Custom Ops

`direct_register_custom_op` — 部分 attention 注册为 PyTorch custom op，便于 compile/graph capture。

## FlashInfer 集成

README 提及 FlashInfer 集成；部分路径与 FlashAttention 并存，以版本/平台为准。

## 性能要点

| 阶段 | 关键 |
|------|------|
| Prefill | FlashAttention varlen，大 matmul |
| Decode | Memory-bound，GQA/MLA 减 KV 读 |
| Graph | CUDA graph 捕获 decode 小 batch |

`--enforce-eager` 禁用 graph 便于调试。

## 与 llama.cpp 对照

| vLLM | llama.cpp |
|------|-----------|
| FlashAttention paged | ggml flash attn / cuBLAS |
| Backend selector | `llama_flash_attn_type` |
| MLA backend | DeepSeek 专用 kernel |

见 `llama.cppDoc` attention 与 KV 章节。

## 调试

- OOM：减 batch 或 seq，查 KV block
- Wrong output：查 dtype、RoPE、GQA head 数
- Fallback：日志中 `Using attention backend: ...`
