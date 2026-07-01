# 05 - 内存分配器

## 1. 模块概述

| 文件 | 行数 | 职责 |
|------|------|------|
| `include/ggml-alloc.h` | ~86 | 公共分配 API |
| `src/ggml-alloc.c` | ~1,248 | 三层分配器实现 |

GGML 有两套内存体系：

1. **Context 级**：bump allocator（`ggml_init` 预分配，见 [03-tensor-graph.md](./03-tensor-graph.md)）
2. **Backend 级**：本文档 — 图级生命周期分配，支持 in-place 复用

---

## 2. 三层分配器

### 2.1 `ggml_tallocr` — 线性 Bump

| 字段 | 说明 |
|------|------|
| `buffer` | 绑定的 backend buffer |
| `offset` | 当前分配位置 |

```c
ggml_tallocr_alloc(talloc, tensor);  // 顺序分配，不可 free
```

用途：单 buffer 内顺序分配权重。

### 2.2 `ggml_dyn_tallocr` — Best-Fit 动态

| 特性 | 说明 |
|------|------|
| 算法 | best-fit 空闲块 + 合并 |
| 支持 | allocate + free |
| 用途 | 图中间张量 in-place 复用 |

关键：维护空闲块链表，节点计算完成后 `free_node` 释放，供后续节点复用。

### 2.3 `ggml_gallocr` — 图级分配器

```c
struct ggml_gallocr {
    ggml_dyn_tallocr * bufts[GGML_VBUFFER_MAX_CHUNKS];  // 每 chunk 一个
    ggml_backend_buffer_t buffers[GGML_VBUFFER_MAX_CHUNKS];
    int n_buffers;
};
```

**核心**：按计算图拓扑序，为每个节点分配/释放内存，最大化复用。

---

## 3. 关键 API

| API | 说明 |
|-----|------|
| `ggml_gallocr_new(n_bufts)` | 创建图分配器 |
| `ggml_gallocr_reserve(galloc, gf)` | 用 worst-case 图预分配 |
| `ggml_gallocr_alloc_graph(galloc, gf)` | 为当前图分配 |
| `ggml_gallocr_free(galloc)` | 释放 |
| `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)` | Context 内所有 tensor 一次性分配 |
| `ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft)` | 同上，返回 size |

---

## 4. `ggml_gallocr_alloc_graph` 流程

```
alloc_graph(galloc, cgraph)
    |
    +-- 拓扑序遍历 nodes[]
    |
    +-- 对每个 node:
    |     +-- allocate_node(node)     # 分配输出 tensor buffer
    |     |     +-- 检查 view/parent 共享
    |     |     +-- 检查 in-place（ggml_op_can_inplace）
    |     +-- [node 计算完成]
    |     +-- free_node(node)          # 释放可复用内存
    |
    +-- OUTPUT flag 的 tensor 永不覆盖
```

### 4.1 In-Place 算子（`ggml_op_can_inplace` L22-50）

可原地执行、输出覆盖输入的 op：

```
ADD, MUL, SCALE, DIAG, SOFT_MAX, ROPE, RMS_NORM,
UNARY(SILU/GELU/...), GLU, ...
```

gallocr 检测到 in-place op 时，输出 tensor 复用输入 buffer，节省内存。

### 4.2 View 与 Parent 共享

- `view_src` 非空：与源 tensor 共享 buffer + offset
- 多个 view 指向同一 parent：共享计数，parent free 时才真正释放

---

## 5. 与调度器协作

```
ggml_backend_sched
    |
    +-- galloc (ggml_gallocr)
    |     +-- bufts[] 对应各 Backend 的 buffer type
    |     +-- vbuffer 支持多 chunk（最多 16）
    |
    +-- split_graph 后，每个 split 独立 alloc_graph
    +-- reserve(worst_case_graph) 预分配最大 buffer
```

llama.cpp 流程：

```
初始化:
    graph_reserve(worst_case) -> gallocr_reserve + sched_reserve

每次 decode:
    sched_alloc_graph(gf)  -> gallocr_alloc_graph
    [若失败] graph_reserve 重算 worst-case
```

---

## 6. 张量 Flag 与分配策略

| Flag | 分配行为 |
|------|----------|
| `INPUT` | 图开头非重叠地址分配 |
| `OUTPUT` | 永不 free，不被 in-place 覆盖 |
| 普通中间节点 | 计算完成后立即 free，内存复用 |

---

## 7. 非显而易见细节

1. **生命周期复用**：类似 arena + free list，中间张量内存可在图内复用，峰值内存远小于所有节点之和
2. **`GGML_VBUFFER_MAX_CHUNKS=16`**：支持跨多种 backend buffer type 分配
3. **`ggml_backend_buffer_get_alloc_size`**：Backend 可报告大于 `ggml_nbytes` 的对齐大小（GPU padding）
4. **reserve vs alloc**：reserve 用 worst-case 图（最大 batch/n_tokens）预分配，避免运行时 realloc
5. **权重加载**：`ggml_backend_alloc_ctx_tensors_from_buft` 一次性为 context 内所有权重量分配 GPU buffer

---

## 8. 内存优化建议

| 场景 | 方法 |
|------|------|
| 减少峰值 | 启用 in-place op；Flash Attention 减少 KV 大小 |
| 避免 realloc | decode 前 `sched_reserve(worst_case_graph)` |
| 多 GPU | 权重按 layer 分配到各 GPU buffer |
| 调试 OOM | 设置 `GGML_SCHED_NO_REALLOC=1` 观察 |

---

## 9. 相关文档

- [04-backend-scheduler.md](./04-backend-scheduler.md) - sched 如何使用 gallocr
- [13-llama-cpp-integration.md](./13-llama-cpp-integration.md) - llama graph_reserve
