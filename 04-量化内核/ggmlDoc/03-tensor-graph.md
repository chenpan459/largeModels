# 03 - 张量模型与计算图

## 1. 模块概述

| 文件 | 行数 | 职责 |
|------|------|------|
| `include/ggml.h` | ~2,863 | 公共 API、枚举、结构体 |
| `src/ggml.c` | ~7,815 | 张量创建、算子工厂、图构建 |
| `src/ggml.cpp` | ~26 | C++ 异常/backtrace 钩子 |
| `src/ggml-impl.h` | — | 内部结构（context、cgraph、object、hash_set） |

## 2. 关键常量（`ggml.h`）

```c
#define GGML_MAX_DIMS      4
#define GGML_MAX_SRC       10      // 每算子最多 10 个输入
#define GGML_MAX_OP_PARAMS 64      // int32_t 数组（非结构体）
#define GGML_MAX_NAME      64
#define GGML_MEM_ALIGN     16      // x86_64；wasm 为 8
```

## 3. `struct ggml_tensor`

```c
struct ggml_tensor {
    enum ggml_type type;
    struct ggml_backend_buffer * buffer;
    int64_t ne[GGML_MAX_DIMS];     // 维度 [d0,d1,d2,d3]
    size_t  nb[GGML_MAX_DIMS];     // stride（字节）
    enum ggml_op op;
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t flags;                 // INPUT/OUTPUT/PARAM/LOSS/COMPUTE
    struct ggml_tensor * src[GGML_MAX_SRC];
    struct ggml_tensor * view_src;
    size_t view_offs;
    void * data;
    char name[GGML_MAX_NAME];
    void * extra;                  // Backend 私有（CUDA extras 等）
};
```

### 张量 Flag

| Flag | 含义 | 调度/分配影响 |
|------|------|---------------|
| `GGML_TENSOR_FLAG_INPUT` | 用户输入 | 默认落最后 Backend（通常 CPU） |
| `GGML_TENSOR_FLAG_OUTPUT` | 输出节点 | gallocr **永不覆盖** |
| `GGML_TENSOR_FLAG_PARAM` | 可训练参数 | ggml-opt 使用 |
| `GGML_TENSOR_FLAG_LOSS` | 损失节点 | 反向传播起点 |
| `GGML_TENSOR_FLAG_COMPUTE` | 中间计算标记 | 调度 hint |

## 4. `enum ggml_type`（42 种）

| 类别 | 类型 |
|------|------|
| 浮点 | F32, F16, BF16 |
| 标准 block | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, **Q1_0** |
| K-quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K（super-block=256） |
| Importance | IQ1_*, IQ2_*, IQ3_*, IQ4_* |
| Ternary | TQ1_0, TQ2_0 |
| FP4 | MXFP4, NVFP4 |

**兼容性原则**：新类型只在 enum 末尾追加；旧 GGUF 仍可读。

### `type_traits[]`（`ggml.c` L621+）

每种类型关联：

| 字段 | 说明 |
|------|------|
| `blck_size` | block 元素数（32 或 256） |
| `type_size` | 单 block 字节数 |
| `is_quantized` | 是否量化 |
| `to_float` | dequant 函数指针 |
| `from_float_ref` | quant 参考实现指针 |

## 5. `enum ggml_op`（88 种）

### LLM 推理

| 算子 | 用途 |
|------|------|
| `MUL_MAT` | 矩阵乘 |
| `MUL_MAT_ID` | MoE |
| `ROPE` | RoPE |
| `RMS_NORM` / `LAYER_NORM` | 归一化 |
| `SOFT_MAX` | Softmax |
| `FLASH_ATTN_EXT` | Flash Attention |
| `GET_ROWS` / `SET_ROWS` | Embedding |
| `GLU` | SwiGLU 等（子类型见 `ggml_glu_op`） |
| `RWKV_WKV6/7` | RWKV |
| `SSM_CONV` / `SSM_SCAN` | Mamba |
| `GATED_DELTA_NET` | DeltaNet |
| `GATED_LINEAR_ATTN` | 门控线性注意力 |

### 形状与数据

`RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`, `CONT`, `CPY`, `DUP`, `CONCAT`, `PAD`, `ROLL`

### 训练

`OPT_STEP_ADAMW`, `OPT_STEP_SGD`, `CROSS_ENTROPY_LOSS`, `build_backward_expand`

### 辅助枚举

- `ggml_unary_op`：22 种（SILU、GELU、TANH…）
- `ggml_glu_op`：SWIGLU、GEGLU、REGLU、SWIGLU_OAI 等
- `ggml_prec`：F32 精度 hint
- `ggml_op_hint`：Hadamard 等

## 6. `struct ggml_cgraph`（`ggml-impl.h` L329+）

```c
struct ggml_cgraph {
    int size, n_nodes, n_leafs;
    struct ggml_tensor ** nodes;      // 计算节点（拓扑序）
    struct ggml_tensor ** leafs;      // 常量/输入/权重
    struct ggml_tensor ** grads;      // 梯度（训练）
    struct ggml_tensor ** grad_accs;  // 梯度累加器
    int32_t * use_counts;             // 引用计数（hash slot 索引）
    struct ggml_hash_set visited_hash_set;
    enum ggml_cgraph_eval_order order;
    uint64_t uid;                     // 图唯一 ID（sched 复用检测）
};
```

`use_counts`：每个 node 被后续节点引用的次数；归零时 gallocr 可 free 其输出 buffer。

## 7. Context Bump Allocator

```c
struct ggml_object {
    size_t offs, size;
    struct ggml_object * next;
    enum ggml_object_type type;  // TENSOR / GRAPH / WORK_BUFFER
};

struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool mem_buffer_owned;
    bool no_alloc;               // true: 不分配 tensor data
    int n_objects;
    struct ggml_object * objects_begin, * objects_end;
};
```

- `ggml_new_object()`：从 `mem_buffer` 顺序分配
- `ggml_reset(ctx)`：重置对象链表，**不释放** mem_buffer
- `no_alloc=true`：只建元数据；data 由 Backend/GGUF 后续绑定

## 8. 关键 API

### 初始化

```c
struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,
    .no_alloc   = false,
};
struct ggml_context * ctx = ggml_init(params);
ggml_free(ctx);
```

### 张量创建

| API | 说明 |
|-----|------|
| `ggml_new_tensor_1d/2d/3d/4d` | 指定维度 |
| `ggml_new_tensor` | 通用 |
| `ggml_view_tensor` / `ggml_view_*` | 零拷贝 view |
| `ggml_cont` | 强制连续化 |
| `ggml_dup` | 复制 |

### 建图

| API | 行号(约) | 说明 |
|-----|----------|------|
| `ggml_visit_parents_graph` | L6920+ | DFS，维护 use_counts |
| `ggml_build_forward_impl` | L6964 | 递归建前向图 |
| `ggml_build_forward_expand` | L6997 | 追加到已有图 |
| `ggml_build_forward_select` | L6983 | 多根图，仅 idx 分支 |
| `ggml_build_backward_expand` | L7001 | 反向（训练） |
| `ggml_new_graph` | L7133 | 创建空图 |
| `ggml_graph_view` | — | 子图 view |
| `ggml_graph_compute` | `ggml-cpu.c` L3308 | CPU 执行 |

### 量化辅助

| API | 说明 |
|-----|------|
| `ggml_quantize_chunk()` | 统一量化入口 |
| `ggml_quantize_requires_imatrix()` | IQ 系列强制 imatrix |
| `ggml_quantize_init(type)` | 惰性初始化 codebook |

## 9. LLM 常用算子示例

```c
struct ggml_tensor * ggml_mul_mat(ctx, w, x);
struct ggml_tensor * ggml_rope(ctx, a, pos, n_dims, mode);
struct ggml_tensor * ggml_rms_norm(ctx, a, eps);
struct ggml_tensor * ggml_flash_attn_ext(ctx, q, k, v, mask, ...);
struct ggml_tensor * ggml_get_rows(ctx, embd, tokens);
struct ggml_tensor * ggml_glu(ctx, a, b, GGML_GLU_OP_SWIGLU);
struct ggml_tensor * ggml_mul_mat_id(ctx, experts, x, ids);
```

## 10. 建图与执行流程

```
1. ggml_init()
2. 创建 weight/input tensor（或 no_alloc + GGUF 绑定）
3. out = ggml_mul_mat(ctx, w, x)       # 只建图
4. gf = ggml_new_graph(ctx)
5. ggml_build_forward_expand(gf, out)  # 拓扑排序 + use_counts
6. ggml_backend_sched_alloc_graph(sched, gf)
7. 填充 input tensor data
8. ggml_backend_sched_graph_compute_async(sched, gf)
9. ggml_backend_sched_synchronize(sched)
10. 读取 output tensor data
```

## 11. In-Place 算子

`ggml_op_can_inplace()`（`ggml-alloc.c` L22-50）列出可原地执行的 op：

ADD, MUL, SCALE, SOFT_MAX, ROPE, RMS_NORM, UNARY(SILU/GELU/…), GLU 等。

gallocr 据此复用 parent buffer，降低峰值内存。

## 12. 非显而易见细节

1. **View 非连续**：`nb[]` 不规则时 GPU kernel 需特殊路径或 `ggml_cont`
2. **4 维上限**：LLM 权重通常 `[rows, cols, 1, 1]`
3. **反向传播**：llama.cpp 推理不用；ggml-opt 训练使用
4. **`GGML_OP_NAME[]`**：调试用算子名称表（`ggml.c` L972+）
5. **graph uid**：sched 检测图结构变化，决定 realloc

## 相关文档

- [04-backend-scheduler.md](./04-backend-scheduler.md)
- [05-memory-alloc.md](./05-memory-alloc.md)
- [06-quantization.md](./06-quantization.md)
