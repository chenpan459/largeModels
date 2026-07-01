# 03 - 张量模型与计算图

## 1. 模块概述

| 文件 | 行数 | 职责 |
|------|------|------|
| `include/ggml.h` | ~2,863 | 公共 API、枚举、结构体 |
| `src/ggml.c` | ~7,815 | 张量创建、算子工厂、图构建 |
| `src/ggml.cpp` | ~26 | C++ 异常/backtrace 钩子 |
| `src/ggml-impl.h` | — | 内部结构（context, cgraph, object） |

---

## 2. 核心数据结构

### 2.1 `struct ggml_tensor`

```c
struct ggml_tensor {
    enum ggml_type type;           // F32, Q4_K, ...
    struct ggml_backend_buffer * buffer;
    int64_t ne[GGML_MAX_DIMS];     // 维度大小 [d0,d1,d2,d3]
    size_t  nb[GGML_MAX_DIMS];     // stride（字节）
    enum ggml_op op;               // 产生此张量的算子
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t flags;                   // INPUT/OUTPUT/PARAM/LOSS
    struct ggml_tensor * src[GGML_MAX_SRC];  // 输入张量（最多 10 个）
    struct ggml_tensor * view_src;  // view 的源张量
    size_t view_offs;
    void * data;                   // 数据指针
    char name[GGML_MAX_NAME];
    void * extra;                  // Backend 私有扩展
};
```

### 2.2 `enum ggml_type`（42 种）

| 类别 | 类型 | 说明 |
|------|------|------|
| 浮点 | F32, F16, BF16 | 标准浮点 |
| 4-bit | Q4_0, Q4_1, Q4_K, Q4_K_S, Q4_K_M, Q4_K_L | block=32 或 super-block |
| 5-bit | Q5_0, Q5_1, Q5_K_* | |
| 8-bit | Q8_0, Q8_1, Q8_K | |
| 2-bit | Q2_K, Q3_K_* | K-quants |
| 1-bit | IQ1_*, IQ2_*, TQ1_0 | Importance/Ternary |
| FP4 | MXFP4, NVFP4 | 新浮点4bit |

**兼容性原则**（`ggml.h` 注释）：新类型只在 enum 末尾追加，保证 GGUF 向后兼容。

### 2.3 `enum ggml_op`（88 种）

分类概览：

| 类别 | 代表算子 |
|------|----------|
| 一元 | `UNARY` (ABS/NEG/SILU/GELU/...) |
| 二元 | `ADD`, `SUB`, `MUL`, `DIV`, `MUL_MAT` |
| 归约 | `SUM`, `SUM_ROWS`, `MEAN`, `ARGMAX` |
| 形状 | `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`, `CONT` |
| 归一化 | `RMS_NORM`, `LAYER_NORM`, `GROUP_NORM` |
| 注意力 | `SOFT_MAX`, `FLASH_ATTN_EXT`, `ROPE` |
| Embedding | `GET_ROWS`, `SET_ROWS`, `ADD_ID` |
| 激活 | `GLU` (SwiGLU/GeGLU/ReGLU) |
| MoE | `MUL_MAT_ID` |
| 卷积 | `CONV_1D`, `CONV_2D`, `IM2COL` |
| 状态空间 | `SSM_CONV`, `SSM_SCAN`, `RWKV_WKV6/7` |
| 扩散 | `TIMESTEP_EMBEDDING`, `GLU` |
| 自定义 | `CUSTOM`, `MAP_CUSTOM*` |

### 2.4 `struct ggml_cgraph`

```c
struct ggml_cgraph {
    int size;                // 容量
    int n_nodes;             // 计算节点数
    int n_leafs;             // 叶子节点（输入/权重）
    struct ggml_tensor ** nodes;   // 拓扑排序的计算节点
    struct ggml_tensor ** leafs;   // 输入和常量
    struct ggml_tensor ** grads;    // 梯度（训练用）
    struct ggml_hash_set visited_hash_set;
    enum ggml_cgraph_eval_order order;
};
```

### 2.5 `struct ggml_context`

```c
struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
    int    n_objects;
    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};
```

Context 是 **bump allocator**：所有 tensor/graph 元数据从 `mem_buffer` 顺序分配，不可单独释放。

---

## 3. 关键 API

### 3.1 初始化

| API | 说明 |
|-----|------|
| `ggml_init(params)` | 创建 Context，预分配 `mem_size` |
| `ggml_free(ctx)` | 释放整个 Context |

```c
struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,
    .mem_buffer = NULL,  // NULL 则内部分配
    .no_alloc   = false,
};
struct ggml_context * ctx = ggml_init(params);
```

### 3.2 张量创建

| API | 说明 |
|-----|------|
| `ggml_new_tensor_1d/2d/3d/4d` | 指定维度创建 |
| `ggml_new_tensor` | 通用创建 |
| `ggml_view_tensor` | 零拷贝 view |
| `ggml_dup` | 复制张量 |

### 3.3 建图

| API | 行号(约) | 说明 |
|-----|----------|------|
| `ggml_build_forward_impl` | L6964 | 递归构建前向图 |
| `ggml_build_forward_expand` | L6998 | 追加节点到已有图 |
| `ggml_new_graph` | L7133 | 创建空图 |
| `ggml_graph_compute` | ggml-cpu.c L3308 | CPU 执行 |

### 3.4 LLM 常用算子

```c
// 矩阵乘（Attention QK^T, FFN）
struct ggml_tensor * ggml_mul_mat(ctx, a, b);

// RoPE 位置编码
struct ggml_tensor * ggml_rope(ctx, a, pos, n_dims, mode);

// RMSNorm
struct ggml_tensor * ggml_rms_norm(ctx, a, eps);

// Flash Attention
struct ggml_tensor * ggml_flash_attn_ext(ctx, q, k, v, mask, ...);

// Embedding lookup
struct ggml_tensor * ggml_get_rows(ctx, embd, tokens);

// SwiGLU
struct ggml_tensor * ggml_glu(ctx, a, b, type);  // type=SWIGLU

// MoE 专家路由
struct ggml_tensor * ggml_mul_mat_id(ctx, as, b, ids);
```

---

## 4. 建图与执行流程

```
1. ggml_init()                         # 分配 Context
2. 创建输入 tensor（weights, tokens）
3. out = ggml_mul_mat(ctx, w, x)       # 只建图，不计算
4. gf = ggml_new_graph(ctx)
5. ggml_build_forward_expand(gf, out)  # 拓扑排序
6. 填充 input tensor data
7. ggml_graph_compute(ctx, gf, n_threads)  # 或 Backend 路径
8. 读取 output tensor data
```

---

## 5. 张量 Flag

| Flag | 含义 | 调度影响 |
|------|------|----------|
| `GGML_TENSOR_FLAG_INPUT` | 用户输入 | 默认落最后一个 Backend（通常 CPU） |
| `GGML_TENSOR_FLAG_OUTPUT` | 输出节点 | gallocr 永不覆盖其内存 |
| `GGML_TENSOR_FLAG_PARAM` | 可训练参数 | 优化器使用 |
| `GGML_TENSOR_FLAG_LOSS` | 损失节点 | 反向传播起点 |

---

## 6. `ggml.cpp`（26 行）

仅注册 `std::terminate` 处理器：未捕获 C++ 异常时打印 backtrace（`GGML_NO_BACKTRACE` 可禁用）。**无张量逻辑**。

---

## 7. 非显而易见细节

1. **View 与 Contiguous**：`ggml_view_*` 共享底层 buffer；非连续 tensor 在 GPU kernel 中需特殊处理
2. **In-place 算子**：`ggml_op_can_inplace()` 列出可原地执行的 op（ADD/MUL/ROPE/RMS_NORM 等），gallocr 据此复用内存
3. **`no_alloc=true`**：Context 只建元数据不分配 tensor data；llama 加载 GGUF 时使用
4. **4 维上限**：`GGML_MAX_DIMS=4`，LLM 权重通常 `[rows, cols, 1, 1]`
5. **反向传播**：`ggml_build_backward_expand` 存在但 llama.cpp 推理不使用

---

## 8. 扩展指南

| 需求 | 修改位置 |
|------|----------|
| 新算子 | `ggml.h` 添加 enum + `ggml.c` 工厂函数 + 各 Backend 实现 |
| 新数据类型 | `ggml_type` enum + `ggml-common.h` block 定义 + `ggml-quants.c` |
| 调试建图 | 设置 `GGML_DEBUG` 环境变量打印图结构 |

---

## 9. 相关文档

- [04-backend-scheduler.md](./04-backend-scheduler.md) - 图如何分配到 Backend
- [05-memory-alloc.md](./05-memory-alloc.md) - 图级内存生命周期
- [06-quantization.md](./06-quantization.md) - 量化类型详解
