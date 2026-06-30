# 17 - Batch 系统与 Micro-batch 拆分

## 1. 概述

Batch 系统是 llama.cpp 连续批处理（Continuous Batching）和多序列并行的基础。它将用户提交的 `llama_batch` 拆分为更小的 `llama_ubatch`（micro-batch），逐个送入计算图执行。

| 文件 | 行数 | 职责 |
|------|------|------|
| `llama-batch.cpp` | 919 | Batch 分配器实现 |
| `llama-batch.h` | 173 | 结构定义 |

---

## 2. 核心数据结构

### 2.1 `llama_batch`（公开 API）

用户面向的 batch 结构，定义在 `include/llama.h`：

```cpp
struct llama_batch {
    int32_t n_tokens;
    llama_token  * token;      // token ID 数组
    float        * embd;       // 或直接 embedding 输入（二选一）
    llama_pos    * pos;        // 位置数组
    int32_t      * n_seq_id;   // 每个 token 的 seq_id 数量
    llama_seq_id ** seq_id;    // 序列 ID 二维数组
    int8_t       * logits;     // 是否输出 logits (0/1)
};
```

**使用方式**：

- 手动填充所有字段
- 或使用 `llama_batch_get_one(tokens, n)` 快速创建单序列 batch
- 或使用 `llama_batch_init()` 预分配

### 2.2 `llama_ubatch`（内部 micro-batch）

```cpp
struct llama_ubatch {
    bool b_equal_seqs;         // 是否等长多 seq 批

    uint32_t n_tokens;         // 本 ubatch token 总数
    uint32_t n_seq_tokens;     // 每个 seq 的 token 数
    uint32_t n_seqs;           // seq 数量
    uint32_t n_seqs_unq;       // 唯一 seq 数
    uint32_t n_pos;            // 位置维度（M-RoPE 等多维位置）

    llama_token    * token;
    float          * embd;
    llama_pos      * pos;
    llama_seq_id   * seq_id;
    int8_t         * output;   // 是否输出

    std::shared_ptr<data_t> data;  // 自有数据时使用
};
```

`llama_ubatch` 是轻量视图，数据可指向原始 batch 或 `shared_ptr` 自有 buffer。

### 2.3 `llama_batch_allocr`

Batch 分配器，负责验证、补全和拆分：

| 成员 | 说明 |
|------|------|
| `batch` | 当前处理的 batch 副本 |
| `seq_pos[]` | 各 seq 当前最大位置 |
| `seq_cpl[][]` | coupled sequences 关系 |
| `seq_idx[]` | seq_id -> 内部索引映射 |
| `seq_set_map` | token -> seq 集合 |
| `out_ids[]` | output token 在原始 batch 中的索引 |
| `debug` | 调试级别 |

---

## 3. init()：Batch 消毒与自动补全

`llama_batch_allocr::init()`（L25-145）是 batch 进入系统的第一站：

### 3.1 输入验证

```
init(batch_inp, vocab, memory, n_embd, n_seq_max, output_all)
  |
  +-- 验证 n_seq_max <= LLAMA_MAX_SEQ
  +-- 验证 token ID 范围 [0, vocab.n_tokens)
  +-- 验证 seq_id 范围 [0, n_seq_max)
```

### 3.2 自动补全缺失字段

| 缺失字段 | 补全策略 |
|----------|----------|
| `n_seq_id` / `seq_id` | 全部分配 seq 0 |
| `pos` | 从 `memory->seq_pos_max(s)+1` 递增 |
| `logits` | 仅最后一个 token 设为 1；`output_all=true` 时全部设为 1 |

### 3.3 构建 seq_set_map

为每个 token 记录其所属的 seq 集合（一个 token 可属于多个 seq，即 coupled sequences）。后续拆分策略依赖此映射。

### 3.4 Coupled Sequences

当同一 token 同时属于多个 seq 时（如 prompt 共享），标记为 coupled。这影响 `split_equal` 的行为。

---

## 4. 三种拆分策略

### 4.1 策略对比

| 策略 | 函数 | 使用场景 | 特点 |
|------|------|----------|------|
| `split_simple` | L474 | 单 stream KV (`n_stream==1`) | 按 token 顺序切，最大 n_ubatch |
| `split_equal` | L508 | 多 stream KV | 各 seq set 同步取 token，保证 stream 对齐 |
| `split_seq` | L613 | 循环模型 (Mamba/RWKV) | 每次一个 seq set |

### 4.2 split_simple（L474）

```
split_simple(n_ubatch)
  |
  +-- 从当前位置顺序取最多 n_ubatch 个 token
  +-- 不限 seq 结构，可跨 seq 边界
  +-- 返回 llama_ubatch
```

**适用**：单 stream KV cache，最常见的 decode 场景。

**示例**：batch 有 10 个 token，`n_ubatch=4`：
- ubatch 0: token[0..3]
- ubatch 1: token[4..7]
- ubatch 2: token[8..9]

### 4.3 split_equal（L508）

```
split_equal(n_ubatch, sequential)
  |
  +-- 识别所有 seq set
  +-- 各 seq set 同步取 n_seq_tokens 个 token
  +-- 保证每个 ubatch 中各 stream 的 token 数相等
  +-- 返回 llama_ubatch (b_equal_seqs=true)
```

**适用**：多 stream KV cache（并行多序列）。

**示例**：2 个 seq，各 5 个 token，`n_ubatch=4`（即每 seq 2 个 token）：
- ubatch 0: seq0[0,1] + seq1[0,1]（4 tokens, 2 seqs, 2 seq_tokens）
- ubatch 1: seq0[2,3] + seq1[2,3]
- ubatch 2: seq0[4] + seq1[4]

### 4.4 split_seq（L613）

```
split_seq(n_ubatch)
  |
  +-- 每次取一个 seq set 的全部 token
  +-- 返回 llama_ubatch
```

**适用**：循环模型，状态不可跨 seq 混合。

---

## 5. KV Cache 中的拆分调用

`llama_kv_cache::init_batch()`（L711-746）选择拆分策略：

```cpp
auto ubatch = n_stream == 1
    ? balloc.split_simple(n_ubatch)
    : balloc.split_equal(n_ubatch, true);
```

循环直到 `ubatch.n_tokens == 0` 或 `get_n_used() < get_n_tokens()`（拆分失败）。

---

## 6. ubatch 构造

### 6.1 ubatch_add（L681）

从 batch 中指定范围的 token 构造 ubatch：

- 复制 token/embd/pos/seq_id/output 指针或数据
- 设置 n_tokens, n_seqs, n_seq_tokens, b_equal_seqs
- 记录 out_ids（output token 在原始 batch 中的索引）

### 6.2 ubatch_reserve（L391）

预留空 ubatch，用于 `graph_reserve()` worst-case 图预留。

---

## 7. Output 映射

### 7.1 out_ids

拆分过程中，`out_ids[]` 记录每个 output token 在**原始 batch** 中的索引。

### 7.2 decode 结束后的重映射

```
decode 完成后:
  output_ids[i] = out_ids 中第 i 个 output 的原始索引
  output_reorder() 按原始顺序重排 logits/embd
```

这保证即使 batch 被拆分为多个 ubatch，最终输出顺序与用户输入一致。

---

## 8. 约束与边界情况

### 8.1 n_batch vs n_ubatch

| 参数 | 含义 | 关系 |
|------|------|------|
| `n_batch` | 逻辑 batch 最大 token 数 | `n_tokens_all <= n_batch` |
| `n_ubatch` | 单次 graph compute 最大 token 数 | `n_ubatch <= n_batch` |

### 8.2 非 Causal Attention

```cpp
GGML_ASSERT(cparams.causal_attn || cparams.n_ubatch >= n_tokens_all);
```

非 causal attention（如 BERT encode）要求整个 batch 在一个 ubatch 中处理。

### 8.3 Backend 采样约束

每个 seq 最多一个 output token（decode L1708-1728）。违反则返回 -1。

### 8.4 Embedding 模式

`output_all=true` 时，所有 token 都必须标记为 output，否则报错。

---

## 9. 连续批处理（Continuous Batching）

Server 中的连续批处理依赖 batch 系统：

```
Request A (seq 0, 3 tokens)  --+
Request B (seq 1, 2 tokens)  --+--> 合并为一个 llama_batch
Request C (seq 2, 1 token)   --+
                                    |
                                    v
                              balloc.init()
                                    |
                                    v
                              split_simple/equal
                                    |
                                    v
                              多个 ubatch 依次 process_ubatch
```

关键特性：

- 不同请求的 token 可在同一 batch 中
- 通过 `seq_id` 隔离 KV cache
- 新请求可动态加入正在处理的 batch

---

## 10. 调试

| 环境变量 | 作用 |
|----------|------|
| `LLAMA_BATCH_DEBUG=1` | 基本拆分日志 |
| `LLAMA_BATCH_DEBUG=2` | 详细 ubatch 内容 (`ubatch_print`) |

---

## 11. 典型使用模式

### 11.1 单 token 生成（最常见）

```c
llama_token token = llama_sampler_sample(sampler, ctx, -1);
struct llama_batch batch = llama_batch_get_one(&token, 1);
llama_decode(ctx, batch);
```

`llama_batch_get_one` 自动设置 pos（递增）和 logits（最后一个 token）。

### 11.2 Prompt 预填充

```c
llama_token tokens[512];
int n = llama_tokenize(vocab, prompt, len, tokens, 512, true, true);
struct llama_batch batch = llama_batch_get_one(tokens, n);
llama_decode(ctx, batch);  // 可能拆分为多个 ubatch
```

### 11.3 多序列并行

```c
struct llama_batch batch = llama_batch_init(total_tokens, 0, n_seqs);
// 手动填充 token, pos, seq_id, logits
llama_decode(ctx, batch);
```

### 11.4 直接 Embedding 输入

```c
struct llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
batch.embd = embedding_data;  // 不用 batch.token
batch.n_tokens = n_tokens;
llama_decode(ctx, batch);
```

---

## 12. 扩展指南

| 需求 | 修改位置 |
|------|----------|
| 新拆分策略 | 在 `llama_batch_allocr` 添加方法 |
| 修改 auto-fill 逻辑 | `init()` 中的补全段 |
| 支持新 pos 格式 | 调整 `n_pos_per_embd` 和 pos 填充 |

---

## 13. 相关文档

- [15-decode-graph-reuse.md](./15-decode-graph-reuse.md) - decode 中 batch 使用
- [16-kv-cache-memory.md](./16-kv-cache-memory.md) - KV cache 与拆分策略选择
- [11-api-reference.md](./11-api-reference.md) - llama_batch C API
- [12-server.md](./12-server.md) - Server 连续批处理
