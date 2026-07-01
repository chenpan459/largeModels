# 05 - KV Cache 与 PagedAttention

## 问题背景

LLM 推理 KV cache 随 batch × seq_len × layers 线性增长。传统实现按 **最大长度预分配** → 显存碎片与浪费。

**PagedAttention**（vLLM 论文）：把 KV 缓存切成固定大小 **block**（如 16 tokens），用 **block table** 映射逻辑 token 位置 → 物理 block，类似 OS 分页。

## 核心概念

| 概念 | 说明 |
|------|------|
| **Block** | 固定 `block_size` 个 token 的 KV 槽位 |
| **Block Table** | 每个 request 的 logical block → physical block id 列表 |
| **Block Pool** | 全局 GPU block 池，分配/回收 |
| **Prefix Cache** | 相同前缀 block 哈希复用 |

## V1 实现路径

```
KVCacheManager (kv_cache_manager.py)
    → BlockPool (block_pool.py)
    → KVCacheBlock / FreeKVCacheBlockQueue
    → kv_cache_utils.py (hash, allocate)
```

### BlockPool

```python
class BlockPool:
    def __init__(self, num_gpu_blocks: int, enable_caching: bool):
        self.blocks: list[KVCacheBlock] = [...]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block: dict[BlockHashType, ...]
```

- **分配**：`free_block_queue.popleft()`
- **Prefix hit**：`get_cached_block(block_hash)`
- **Eviction**：cache 满时从 free queue 淘汰（LRU 类策略）

### CacheConfig

`config.py` — `CacheConfig`：

| 字段 | 默认/典型 | 含义 |
|------|-----------|------|
| `block_size` | 16 | 每 block token 数 |
| `gpu_memory_utilization` | 0.9 | KV 可用显存比例 |
| `enable_prefix_caching` | false | 前缀缓存 |
| `swap_space` | GiB | CPU swap |
| `cache_dtype` | auto | KV 存储 dtype |

Block 数量在 **EngineCore 启动时 profile**：

```python
available_gpu_memory = self.model_executor.determine_available_memory()
kv_cache_configs = get_kv_cache_config(...)
num_gpu_blocks = kv_cache_configs[0].num_blocks
```

## Attention 层中的 KV

`attention/layer.py` — `Attention` 模块：

1. 将 K/V 写入 paged KV cache（通过 backend）
2. 执行 MHA/GQA/MQA/MLA
3. 返回 output

Backend 负责 **block table 索引** 下的 flash attention。

## KVCacheSpec（V1）

`v1/kv_cache_interface.py` — 不同层类型：

- `FullAttentionSpec`
- `SlidingWindowSpec`
- 支持 hybrid model（部分层 sliding window）

`GPUModelRunner` 绑定 physical KV tensors：`bind_kv_cache()`。

## Prefix Caching

启用后：

1. 对 full block 的 token 内容算 hash
2. `cache_full_blocks()` 注册
3. 新 request 相同前缀 → 直接引用 block，**跳过 prefill 计算**

官方称 V1 **zero-overhead** prefix caching 相对 V0 改进。

## PagedAttention vs 连续 KV

| | 连续 buffer | PagedAttention |
|---|-------------|----------------|
| 分配 | max_len × batch | 按需 block |
| 碎片 | 高 | 低 |
| 内核 | 简单 | block table + FA |
| 共享前缀 | 难 | 自然支持 |

## 与 llama.cpp KV 对照

| vLLM | llama.cpp |
|------|-----------|
| `block_size` | `n_ctx` slot / unified KV |
| BlockPool | `llama_kv_cache` cells |
| Prefix cache | cache reuse / slot prompt |

详见 `llama.cppDoc/16-kv-cache-memory.md`。

## 调优建议

- 增大 `gpu_memory_utilization` → 更多 block → 更高并发
- 减小 `block_size` → 粒度细、表项多；增大 → 相反
- 长对话 + 固定 system prompt → 开 `enable_prefix_caching`
- OOM → 降 `max_num_seqs` 或 `max_model_len`

## 论文

Kwon et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention**, SOSP 2023.
