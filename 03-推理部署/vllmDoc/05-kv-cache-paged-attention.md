# 05 - KV Cache 与 PagedAttention

## 问题背景

LLM 推理 KV cache 随 `batch × seq_len × layers` 线性增长。传统实现按 **最大长度预分配** → 显存碎片与浪费。

**PagedAttention**（vLLM 论文）：把 KV 缓存切成固定大小 **block**（默认 16 tokens），用 **block table** 映射逻辑 token 位置 → 物理 block，类似 OS 虚拟内存分页。

## 核心概念

| 概念 | 说明 |
|------|------|
| **Block** | 固定 `block_size` 个 token 的 KV 槽位（K/V 各一份） |
| **Block Table** | 每个 request 的 logical block index → physical block id |
| **Block Pool** | 全局 GPU block 池，分配/回收/前缀共享 |
| **Prefix Cache** | 对 full block 内容 hash，相同前缀 block 复用 |
| **KVCacheSpec** | 层类型规格（FullAttention / SlidingWindow） |

## V1 模块架构

```
EngineCore._initialize_kv_caches()
  → executor.get_kv_cache_specs()
  → determine_available_memory()  # profile
  → get_kv_cache_config()
  → unify_kv_cache_configs()
  → initialize_from_config()      # 分配 physical tensors
  → bind_kv_cache()               # 绑定到 Attention 层

运行时：
Scheduler → KVCacheManager.allocate_slots()
         → BlockPool.get_new_blocks() / get_cached_block()
Worker   → BlockTable.append_row() / add_row()
         → Attention backend 读 block_table tensor
```

### 关键文件

| 组件 | 路径 |
|------|------|
| `KVCacheManager` | `v1/core/kv_cache_manager.py` |
| `BlockPool` | `v1/core/block_pool.py` |
| `KVCacheBlock`、hash | `v1/core/kv_cache_utils.py` |
| Sliding window | `v1/core/specialized_manager.py` |
| 接口定义 | `v1/kv_cache_interface.py` |
| Worker block table | `v1/worker/block_table.py` |
| Tensor 绑定 | `v1/utils.py` → `bind_kv_cache()` |
| V0 cache engine | `worker/cache_engine.py` |

## BlockPool

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching):
        self.blocks: list[KVCacheBlock] = [...]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block: dict[BlockHashType, KVCacheBlock]
        self.null_block  # 占位 block
```

| 操作 | 方法 |
|------|------|
| 分配新 block | `free_block_queue.popleft()` |
| Prefix hit | `get_cached_block(block_hash)` |
| 注册 full block | `cache_full_blocks()` |
| 释放 | 归还 free queue；cache 中 block 引用计数减 |

### Block 预分配（Preallocation）

`KVCacheManager` 默认 `num_preallocate_tokens = 64`（4 个 block @ block_size=16）：

- 减少每 step 频繁 allocate 的开销
- 与 spec decode lookahead token 分配区分

## KVCacheManager

```python
class KVCacheManager:
    req_to_blocks: dict[str, list[KVCacheBlock]]
    req_to_block_hashes: dict[str, list[BlockHashType]]
```

### allocate_slots(request, num_tokens, num_lookahead_tokens)

1. 计算需要的新 block 数
2. 从 BlockPool 获取 physical blocks
3. 更新 request 的 block 列表
4. 返回新 block ids（供 SchedulerOutput）

失败返回 `None` → 触发 Scheduler 抢占。

### get_computed_blocks() — Prefix Cache Lookup

1. 对已有 prompt block 链计算 hash
2. `find_longest_cache_hit()` 找最长匹配前缀
3. 命中 block 的 `num_computed_tokens` 可直接跳过 prefill

**禁用条件**：

- `enable_prefix_caching=False`
- Request 需要 prompt logprobs（`kv_cache_manager.py:129-131`）

### free(request)

释放 request 全部 block 回 pool；若 block 在 prefix cache 中则减引用。

## BlockTable（Worker 侧）

`v1/worker/block_table.py`：

```python
# CPU/GPU 双缓冲
block_table: Tensor  # [max_num_reqs, max_num_blocks_per_req], int32
```

| 方法 | 作用 |
|------|------|
| `add_row(block_ids)` | 新 request 写入 block table |
| `append_row(req_index, new_block_ids)` | 追加 block |
| `replace_row(req_index, block_ids)` | 抢占恢复时整行替换 |
| `commit()` | CPU → GPU 同步 |

Attention kernel 通过 block table 索引 physical KV tensor 中的 block。

## Physical KV Tensor 布局

V1 FlashAttention backend（`v1/attention/backends/flash_attn.py`）：

```python
# get_kv_cache_shape 返回
(2, num_blocks, block_size, num_kv_heads, head_size)
# 2 = K and V
```

`bind_kv_cache()` 将各层 KV tensor 绑定到 `Attention` 模块的 `kv_cache` 属性。

## KVCacheSpec 与 Hybrid Model

`v1/kv_cache_interface.py`：

| Spec | 用途 |
|------|------|
| `FullAttentionSpec` | 标准 MHA/GQA |
| `SlidingWindowSpec` | 滑动窗口层（如 Mistral） |

`specialized_manager.py` 管理 sliding window 层的 block 回收策略。

**V1 限制**：`len(kv_cache_groups) == 1`（`kv_cache_manager.py:31-33`），即当前仅支持单一 KV cache 组；复杂 hybrid 模型支持在演进中。

## Prefix Caching 详解

启用后（V1 默认 `enable_prefix_caching=True`）：

```
1. Block 写满 block_size 个 token → compute_block_hash()
2. cache_full_blocks() 注册到 cached_block_hash_to_block
3. 新 request 相同前缀 → get_computed_blocks() 命中
4. num_computed_tokens 跳过命中部分 → 节省 prefill 算力
```

### Hash 算法

`prefix_caching_hash_algo`：

- `builtin`（默认）：快速内置 hash
- `sha256`：更强碰撞抵抗

多模态：`generate_block_hash_extra_keys()`（`block_pool.py`）将 mm feature hash 纳入 block hash。

### Cascade Attention

Scheduler 输出 `num_common_prefix_blocks`：batch 内多 request 共享的公共前缀 block 数。Attention backend 可用 cascade attention 优化共享前缀的 attention 计算。

### 统计

`PrefixCacheStats`（`v1/metrics/stats.py`）：queries、hits、hit rate。

API：`POST /reset_prefix_cache`（dev 模式）清空 cache。

## CacheConfig

`config.py` → `CacheConfig`：

| 字段 | 默认/典型 | 含义 |
|------|-----------|------|
| `block_size` | 16（FlashMLA 强制 64） | 每 block token 数 |
| `gpu_memory_utilization` | 0.9 | KV 可用显存比例 |
| `enable_prefix_caching` | V1 默认 True | 前缀缓存 |
| `swap_space` | 4 GiB | CPU swap（**V0 为主**） |
| `cache_dtype` | auto | KV 存储 dtype（含 fp8） |

Block 总数在 **EngineCore 启动时 profile** 决定：

```python
available_gpu_memory = executor.determine_available_memory()
kv_cache_configs = get_kv_cache_config(vllm_config, available_gpu_memory, kv_cache_specs)
num_gpu_blocks = kv_cache_configs[0].num_blocks
```

Profile 流程：加载模型 → 跑一次 dummy forward → 剩余显存 × `gpu_memory_utilization` → 除以 per-block 字节数。

## Attention 层中的 KV

`attention/layer.py` → `Attention` 模块：

1. Q/K/V projection
2. 将 K/V 写入 paged KV cache（backend `advance_step` 或 forward 内 write）
3. 执行 MHA/GQA/MQA/MLA attention
4. 返回 output

Backend 负责在 block table 索引下调用 FlashAttention paged kernel。

## PagedAttention vs 连续 KV

| | 连续 buffer | PagedAttention |
|---|-------------|----------------|
| 分配 | max_len × batch 预分配 | 按需 block |
| 碎片 | 高 | 低 |
| 内核 | 简单 contiguous | block table + FA varlen |
| 共享前缀 | 难 | hash + 引用计数 |
| 抢占 | 需 copy/swap | free block 即可 |
| 动态 batch | 需 padding | 自然支持 |

## FP8 KV Cache

`cache_dtype=fp8` 时：

- V0：需 FlashInfer attention backend
- V1：需 FlashAttention FP8 支持（oracle 检查 `is_fa_fp8_supported()`）
- 显存减半，略有精度损失

## 与 llama.cpp KV 对照

| vLLM | llama.cpp |
|------|-----------|
| `block_size=16` | unified KV cells |
| BlockPool | `llama_kv_cache` slot pool |
| BlockTable | cell index mapping |
| Prefix cache | `--cache-reuse` / slot prompt |
| PagedAttention kernel | ggml flash attn |

详见 `llama.cppDoc/16-kv-cache-memory.md`。

## 调优建议

| 目标 | 手段 |
|------|------|
| 更高并发 | ↑ `gpu_memory_utilization`、↓ `max_model_len` |
| 前缀加速 | 开 `enable_prefix_caching` + 固定 system prompt |
| 长 prompt 不阻塞 | chunked prefill（V1 默认开） |
| OOM | ↓ `max_num_seqs`、↓ `gpu_memory_utilization` |
| 调试 KV | `VLLM_LOGGING_LEVEL=DEBUG` 看 block 分配日志 |

## 论文

Kwon et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention**, SOSP 2023.

## 关键源码行号

| 主题 | 位置 |
|------|------|
| KV 初始化 | `v1/engine/core.py:124-165` |
| allocate_slots | `v1/core/kv_cache_manager.py` |
| Block hash | `v1/core/kv_cache_utils.py` |
| BlockTable | `v1/worker/block_table.py` |
| KV shape | `v1/attention/backends/flash_attn.py` |
