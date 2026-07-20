# 16 - KV Cache 与 Memory 系统

## 1. 概述

llama.cpp 的 Memory 系统为不同模型架构提供统一的序列状态管理抽象。标准 Transformer 使用 KV Cache，Mamba/RWKV 使用循环状态，混合架构（Jamba、Granite Hybrid）则组合两者。

| 文件 | 行数 | 职责 |
|------|------|------|
| `llama-memory.h` | - | 抽象接口定义 |
| `llama-kv-cache.cpp` | 2,632 | 标准 Transformer KV cache |
| `llama-kv-cache.h` | 431 | KV cache 类声明 |
| `llama-kv-cells.h` | 535 | Cell 元数据、defrag、shift |
| `llama-kv-cache-iswa.cpp` | 342 | Interleaved SWA 双 cache |
| `llama-kv-cache-dsa.cpp` | 261 | DeepSeek32 DSA 双 cache |
| `llama-kv-cache-dsa.h` | 139 | DSA cache 声明 |
| `llama-memory-recurrent.cpp` | 1,262 | Mamba/RWKV 循环状态 |
| `llama-memory-hybrid.cpp` | 279 | Attn KV + Recurrent 混合 |
| `llama-memory-hybrid-iswa.cpp` | 285 | Hybrid + ISWA |
| `llama-memory.cpp` | 59 | status 组合辅助 |
| `llama-model.cpp` | - | `create_memory()` 工厂 (L2003-2231) |

---

## 2. 抽象接口

### 2.1 `llama_memory_i`（llama-memory.h）

所有 Memory 后端实现的统一接口：

| 方法 | 说明 |
|------|------|
| `init_batch(balloc, n_ubatch, embd_all)` | 为 batch 分配 slot，返回 context |
| `init_full()` | worst-case 预留（graph_reserve 用） |
| `init_update(lctx, optimize)` | shift/defrag 维护 |
| `seq_rm/cp/keep/add/div` | 序列 CRUD |
| `seq_pos_min/max` | 查询序列位置范围 |
| `state_write/read` | 状态序列化 |
| `memory_breakdown()` | 各 backend buffer 内存占用 |
| `get_can_shift()` | 是否支持 KV shift |

### 2.2 `llama_memory_context_i`

单次 batch 处理的迭代器：

| 方法 | 说明 |
|------|------|
| `next()` | 推进到下一个 ubatch |
| `apply()` | 真正写入 cells / 更新 head |
| `get_ubatch()` | 当前 micro-batch |
| `get_status()` | 分配状态 |

### 2.3 `llama_memory_status`

| 值 | 含义 |
|----|------|
| `SUCCESS` | 分配成功 |
| `NO_UPDATE` | 无需更新（init_update 路径） |
| `FAILED_PREPARE` | 找不到 slot（KV 满） |
| `FAILED_COMPUTE` | 计算失败 |

---

## 3. 标准 KV Cache (`llama_kv_cache`)

### 3.1 核心结构

#### `llama_kv_cache`

| 成员 | 说明 |
|------|------|
| `layers` | `vector<kv_layer>`，每层 K/V tensor |
| `cells` | `llama_kv_cells`，cell 元数据管理 |
| `n_stream` | 并行 stream 数（多序列批处理） |
| `v_heads` | 各 stream 的环形搜索起点 |
| `sc_info` | pending stream copy 信息 |

#### `kv_layer`

| 字段 | 说明 |
|------|------|
| `il` | 层索引 |
| `k`, `v` | K/V cache tensor |
| `k_stream[]`, `v_stream[]` | 多 stream 视图 |

#### `slot_info`

| 字段 | 说明 |
|------|------|
| `s0`, `s1` | stream 范围 |
| `strm[]` | 各 stream 索引 |
| `idxs[][]` | 各 stream 分配的 cell 索引 |

### 3.2 init_batch 流程（L711-746）

```
init_batch(balloc, n_ubatch, embd_all)
  |
  +-- balloc.split_reset()
  +-- while:
  |     ubatch = n_stream==1 ? split_simple(n_ubatch)
  |                          : split_equal(n_ubatch, true)
  |     if ubatch.n_tokens == 0: break
  |     ubatches.push_back(ubatch)
  |
  +-- if balloc.get_n_used() < balloc.get_n_tokens(): FAILED
  +-- sinfos = prepare(ubatches)
  +-- if sinfos.empty(): FAILED
  |
  +-- return llama_kv_cache_context(sinfos, ubatches)
```

### 3.3 prepare：事务性 slot 分配（L760-798）

```
prepare(ubatches)
  |
  +-- for each ubatch:
  |     sinfo = find_slot(ubatch, dry_run=true)   // 仅模拟
  |     if empty: success = false; break
  |     保存 cells 快照到 states[]
  |
  +-- if success:
  |     for each ubatch: find_slot(ubatch, dry_run=false)  // 真正分配
  |     return sinfos
  +-- else:
        恢复所有 cells 快照
        return empty
```

**关键设计**：先模拟分配并保存快照，全部成功才 commit。任一 ubatch 失败则回滚所有 cells 状态。

### 3.4 find_slot：环形 buffer 搜索（L907）

在 KV cache 的环形 buffer 中为 ubatch 找连续空槽：

- 从 `v_heads[stream]` 起点开始搜索
- 检查 cells 是否被其他 seq 占用
- 支持 `cont` 模式（连续分配 vs 分散）
- 更新 `v_heads` 加速下次搜索

### 3.5 llama_kv_cache_context

包装当前 batch 的 KV 视图：

| 方法 | 说明 |
|------|------|
| `get_n_kv()` | 启发式有效 KV 长度（影响 attention mask 维度） |
| `apply()` | 写入 cells 元数据 + 更新 head |
| `next()` | 推进到下一 ubatch |
| `get_k(il)`, `get_v(il)` | 图构建时获取 K/V tensor |
| `cpy_k/cpy_v` | stream copy 操作 |

### 3.6 Cell 管理（llama-kv-cells.h）

`llama_kv_cells` 管理每个 cache cell 的元数据：

| 操作 | 说明 |
|------|------|
| `seq_rm` | 删除序列的 cell 占用 |
| `seq_cp` | 复制序列 cell |
| `seq_add/div` | 位置偏移/缩放 |
| `defrag` | 碎片整理，移动 cell 使空间连续 |
| `shift` | KV cache 左移（context 溢出时） |

---

## 4. KV Cache 变体

### 4.1 ISWA - Interleaved Sliding Window Attention

**文件**: `llama-kv-cache-iswa.cpp`（342 行）

维护 **两个** KV cache 实例：

| Cache | 用途 |
|-------|------|
| 全上下文 cache | 标准 attention |
| SWA cache | 滑动窗口 attention |

适用于 Gemma 2/3、Mistral 等使用 SWA 的模型。Graph input 使用 `llm_graph_input_attn_kv_iswa`。

### 4.2 DSA - DeepSeek Sparse Attention

**文件**: `llama-kv-cache-dsa.cpp`（261 行）、`llama-kv-cache-dsa.h`（139 行）

DeepSeek V3.2 使用 **两个** KV cache 实例：

```cpp
class llama_kv_cache_dsa : public llama_memory_i {
    std::unique_ptr<llama_kv_cache> kv_mla;  // MLA key cache
    std::unique_ptr<llama_kv_cache> kv_lid;  // Lightning Indexer key cache
};
```

| Cache | 存储内容 |
|-------|----------|
| `kv_mla` | 标准 MLA key tensors |
| `kv_lid` | Lightning Indexer key tensors |

`llama_kv_cache_dsa_context` 同时持有两个 sub-context，batch 处理时同步推进。

Graph input 使用 `llm_graph_input_attn_k_dsa`。

### 4.3 Recurrent Memory

**文件**: `llama-memory-recurrent.cpp`（1,262 行）

用于 Mamba、RWKV 等无 KV cache 的循环模型：

- 存储每层的循环状态（SSM state / RWKV channel state）
- 使用 `llm_graph_input_rs` 填充状态
- batch 拆分使用 `split_seq`（每 ubatch 一个 seq set）

### 4.4 Hybrid Memory

**文件**: `llama-memory-hybrid.cpp`（279 行）

Attn 层用 KV cache，Recurrent 层用循环状态：

```cpp
class llama_memory_hybrid : public llama_memory_i {
    llama_kv_cache * kv;           // attention 层
    llama_memory_recurrent * rs;   // recurrent 层
    layer_filter_cb filter_attn;   // 哪些层用 attn
    layer_filter_cb filter_recr;   // 哪些层用 recurrent
};
```

适用于 Jamba、Falcon-H1、Nemotron-H、Qwen3.5 等混合架构。

### 4.5 Hybrid ISWA

**文件**: `llama-memory-hybrid-iswa.cpp`（285 行）

Hybrid + Interleaved SWA 的组合，用于带 SWA 的混合架构。

---

## 5. Memory 工厂决策树

`llama_model::create_memory()`（L2003-2231）：

```
create_memory(params, cparams)
  |
  +-- BERT/LLADA/DREAM/Embedding 等 encode-only -> nullptr
  |
  +-- DEEPSEEK32 -> llama_kv_cache_dsa
  |
  +-- llm_arch_is_recurrent(arch) -> llama_memory_recurrent
  |
  +-- llm_arch_is_hybrid(arch) && !mtp_on_hybrid_qwen35:
  |     +-- swa_type != NONE -> llama_memory_hybrid_iswa
  |     +-- else -> llama_memory_hybrid (带 arch 特定 layer filter)
  |
  +-- default -> llama_kv_cache
        +-- GEMMA3N/4: layer KV reuse callback
        +-- ISWA arch: llama_kv_cache_iswa
        +-- layer filter / share 配置
```

### 5.1 特殊分支说明

| 条件 | 行为 |
|------|------|
| Encode-only 模型 | 返回 `nullptr`，decode 回退到 encode |
| Qwen3.5 MTP context | 强制 plain KV（非 hybrid wrapper） |
| Falcon-H1 | 所有层同时走 attn + recurrent filter |
| Nemotron-H | 按 `is_recr(il)` 和 `n_ff(il)` 分流 |
| Qwen3.5 | 按 `is_recr(il)` 分流 |

---

## 6. KV Cache 参数

创建 KV cache 时的关键参数（来自 `llama_cparams`）：

| 参数 | 说明 |
|------|------|
| `n_ctx_seq` | 每序列最大 context 长度 |
| `n_seq_max` | 最大并行序列数 |
| `type_k`, `type_v` | K/V cache 数据类型（可量化） |
| `v_trans` | V cache 是否转置（`!flash_attn`） |
| `offload_kqv` | KV cache GPU offload |
| `kv_unified` | 统一 KV cache（所有 seq 共享 buffer） |
| `n_swa` | SWA 窗口大小 |
| `swa_type` | SWA 类型 |
| `swa_full` | 是否维护全上下文 SWA cache |

---

## 7. 序列操作

### 7.1 常用 API

```c
llama_kv_cache_clear(ctx);                              // 清空全部
llama_kv_cache_seq_rm(ctx, seq_id, p0, p1);            // 删除序列片段
llama_kv_cache_seq_cp(ctx, src, dst, p0, p1);          // 复制序列 (fork)
llama_kv_cache_seq_keep(ctx, seq_id);                   // 仅保留指定序列
llama_kv_cache_seq_add(ctx, seq_id, p0, p1, delta);    // 位置偏移
```

### 7.2 Server 中的使用

llama-server 的 slot 管理依赖这些 API：

- 新对话：`seq_rm` 清理旧 slot
- Fork/分支：`seq_cp` 复制 KV 状态
- Context shift：超出 n_ctx 时 `seq_add` 或 defrag

---

## 8. 状态序列化

```
llama_state_save_file(ctx, path, tokens, n_tokens)
llama_state_load_file(ctx, path, tokens_out, ...)
```

内部调用 `memory->state_write/read()`，序列化：

- KV cache / 循环状态数据
- Cell 元数据
- 序列位置信息

支持 per-seq 保存（`seq_id` 参数）和 flags 控制。

---

## 9. 内存优化

| 技术 | 说明 | 配置 |
|------|------|------|
| KV 量化 | K/V 用 Q8_0 等低精度 | `--cache-type-k q8_0` |
| Flash Attention | 减少 V cache 内存 | `-fa on` |
| Defrag | 碎片整理释放空间 | 自动（KV 满时） |
| Shift | 左移过期 KV | `get_can_shift()` 为 true 时 |
| Unified KV | 多 seq 共享 buffer | `kv_unified=true` |
| Layer filter | 仅部分层建 KV | MoE/特殊架构 |

---

## 10. 非 obvious 细节

1. **`v_heads` 非 KV 状态**：仅加速 `find_slot` 环形搜索，不代表实际 KV 内容。
2. **多 stream 强制 `split_equal`**：`n_stream > 1` 时 batch 必须用等长 seq 拆分，保证 attention stream 对齐。
3. **`attn_rot_k/v`**：可选 Hadamard 旋转，通过 `LLAMA_ATTN_ROT_DISABLE` 禁用。
4. **`get_n_kv()` 启发式**：未填满 cache 时返回实际使用量而非 `n_ctx`，影响 attention mask 裁剪。
5. **MTP on Qwen3.5**：MTP head 是纯 attention，hybrid 模型的 MTP context 用 plain KV 而非 hybrid wrapper。

---

## 11. 环境变量

| 变量 | 作用 |
|------|------|
| `LLAMA_ATTN_ROT_DISABLE` | 禁用 K/V Hadamard 旋转 |

---

## 12. 扩展指南

| 需求 | 步骤 |
|------|------|
| 新 Memory 类型 | 实现 `llama_memory_i` + `llama_memory_context_i` |
| 新 KV 变体 | 参考 ISWA/DSA 模式，组合多个 `llama_kv_cache` |
| 注册到工厂 | 在 `create_memory()` switch 中添加分支 |
| 新 graph input | 实现 `llm_graph_input_attn_*` 子类 |

---

## 13. 相关文档

- [15-Decode流程与Graph复用.md](./15-Decode流程与Graph复用.md) - decode 中 memory 调用
- [17-Batch与Micro-batch.md](./17-Batch与Micro-batch.md) - batch 拆分与 KV 交互
- [04-模型架构层.md](./04-模型架构层.md) - 各架构 memory 需求
- [03-libllama核心库.md](./03-libllama核心库.md) - libllama 核心概览
