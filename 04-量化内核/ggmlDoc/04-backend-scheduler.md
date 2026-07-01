# 04 - Backend 抽象与调度器

## 1. 模块概述

| 文件 | 行数 | 职责 |
|------|------|------|
| `include/ggml-backend.h` | ~435 | 公共 Backend API |
| `src/ggml-backend.cpp` | ~2,371 | Backend 抽象、调度器实现 |
| `src/ggml-backend-reg.cpp` | ~586 | Backend 注册表 |
| `src/ggml-backend-meta.cpp` | ~2,263 | 张量并行 Meta Backend |
| `src/ggml-backend-dl.cpp` | — | 动态加载 `.so` 插件 |
| `src/ggml-backend-impl.h` | — | 内部接口结构 |

---

## 2. 核心抽象

### 2.1 四层接口（`ggml-backend-impl.h`）

```
ggml_backend_buffer_type_i    # 描述内存类型（CPU RAM / GPU VRAM）
    |
    v
ggml_backend_buffer_i         # 分配/释放 buffer，tensor init/copy
    |
    v
ggml_backend_i                # graph_compute, async, event
    |
    v
ggml_backend_dev_i            # 设备：supports_op, supports_buft, offload_op
    |
    v
ggml_backend_reg_i            # 注册名、设备枚举、proc_address
```

### 2.2 Buffer 用途

| 枚举 | 含义 | 调度影响 |
|------|------|----------|
| `GGML_BACKEND_BUFFER_USAGE_ANY` | 通用 | — |
| `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` | 模型权重 | 调度器优先同 Backend 计算 |
| `GGML_BACKEND_BUFFER_USAGE_COMPUTE` | 中间激活 | 可在 GPU 上分配 |

### 2.3 关键类型

```c
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct ggml_backend_buffer     * ggml_backend_buffer_t;
typedef struct ggml_backend            * ggml_backend_t;
typedef struct ggml_backend_device       * ggml_backend_dev_t;
typedef struct ggml_backend_reg          * ggml_backend_reg_t;
typedef struct ggml_backend_sched        * ggml_backend_sched_t;
```

---

## 3. Backend 注册表

### 3.1 静态注册（`ggml-backend-reg.cpp` L115-167）

构造函数中 `#ifdef GGML_USE_*` 条件注册：

```
CUDA -> Metal -> Vulkan -> SYCL -> HIP -> CPU -> ...
```

**CPU 在末尾注册**，作为最低优先级兜底 Backend。

### 3.2 动态加载（`GGML_BACKEND_DL`）

```c
ggml_backend_load("libggml-cuda.so");   // dlopen
ggml_backend_load_all();                 // 扫描可执行文件目录
ggml_backend_init_best();                // 优先 GPU，否则 CPU
```

### 3.3 设备枚举

```c
ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
int n_devs = ggml_backend_reg_dev_count(reg);
ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
```

---

## 4. 调度器 `ggml_backend_sched`

### 4.1 结构（`ggml-backend.cpp` L774-828）

| 字段 | 说明 |
|------|------|
| `backends[]` | Backend 优先级列表（index 小 = 高优先级） |
| `galloc` | 图级分配器 `ggml_gallocr` |
| `hash_set` + `hv_tensor_backend_ids` | 张量 -> Backend ID 映射 |
| `hv_tensor_copies` | 跨 Backend 拷贝副本（pipeline 并行） |
| `splits[]` | 图切分后的执行片段 |
| `n_copies` | 流水线副本数（默认 4，`GGML_SCHED_MAX_COPIES`） |
| `op_offload` | 是否允许 GPU offload 小 batch |

### 4.2 关键函数

| 函数 | 行号(约) | 职责 |
|------|----------|------|
| `ggml_backend_sched_new` | L1727 | 创建调度器 |
| `ggml_backend_sched_split_graph` | L1014 | 3-pass 分配 Backend + 切分 |
| `ggml_backend_sched_compute_splits` | L1541 | 跨设备 copy + 执行 |
| `ggml_backend_sched_graph_compute` | — | 对外入口（同步） |
| `ggml_backend_sched_graph_compute_async` | — | 异步入口（llama.cpp 使用） |
| `ggml_backend_sched_reserve` | — | worst-case 图预留 buffer |
| `ggml_backend_sched_reset` | — | 重置状态（新图前） |
| `ggml_backend_sched_synchronize` | — | 等待所有 async 完成 |

### 4.3 三 Pass 调度算法

```
Pass 1: 已分配 buffer 的张量/leaf 绑定 Backend
    - 权重在 GPU buffer -> 绑定 GPU Backend
    - INPUT flag -> 通常最后一个 Backend

Pass 2: 相邻节点扩展同一 GPU Backend
    - 向前/向后传播 Backend  assignment
    - CPU 为最低优先级，除非权重在 CPU

Pass 3: 未分配节点
    - 按 supports_op + buffer 兼容性填充
    - 不支持则插入 copy 节点
```

### 4.4 图切分与 Pipeline

```
完整 cgraph
    |
    v
split_graph -> splits[0..n_splits]
    |
    +-- split 0: Backend A (nodes 0..k)
    +-- [copy tensor X: A -> B]
    +-- split 1: Backend B (nodes k+1..m)
    +-- ...
    |
    v
compute_splits (pipeline: n_copies 个 buffer 副本轮转)
```

---

## 5. Meta Backend（张量并行）

`ggml-backend-meta.cpp`（~2,263 行）：

- 包装多个 "simple" 设备，实现 **张量并行**
- `ggml_backend_meta_split_state`：按 axis 切分 Q/K/V
- llama.cpp 多 GPU `--tensor-split` 走此路径

---

## 6. 设备能力查询

```c
// 设备是否支持某算子
bool ggml_backend_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op);

// 设备是否支持某 buffer 类型
bool ggml_backend_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

// 是否应 offload 到该设备（Metal 用于小 batch 阈值判断）
bool ggml_backend_dev_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op);
```

---

## 7. llama.cpp 中的使用

```
llama_model 加载:
    ggml_backend_reg -> 选 devices
    ggml_backend_sched_new(backends, ...)
    ggml_backend_alloc_ctx_tensors_from_buft()  # 权重分配到 GPU

llama_context decode:
    ggml_backend_sched_reset(sched)
    build_graph -> ggml_cgraph
    ggml_backend_sched_alloc_graph(sched, gf)
    ggml_backend_sched_graph_compute_async(sched, gf)
    ggml_backend_sched_synchronize(sched)  # pipeline parallel 复用前
```

---

## 8. 非显而易见细节

1. **MoE 拷贝优化**（L1576+）：`MUL_MAT_ID` 只拷贝用到的 expert slice，非整权重
2. **Pipeline 同步**：复用图前必须 `sched_synchronize`，否则 input tensor 可能被覆盖
3. **INPUT 默认 CPU**：`GGML_TENSOR_FLAG_INPUT` 张量默认在最后一个 Backend（通常 CPU）
4. **supports_op 严格性**：CUDA 的 `supports_op` 检查 tensor 是否在对应 GPU；split buffer 仅支持 `MUL_MAT`
5. **`GGML_SCHED_NO_REALLOC`**：调试环境变量，禁止调度器 realloc

---

## 9. 相关文档

- [05-memory-alloc.md](./05-memory-alloc.md) - sched 使用的 gallocr
- [08-backend-cpu.md](./08-backend-cpu.md) - CPU supports_op（全 op 兜底）
- [09-backend-gpu.md](./09-backend-gpu.md) - CUDA/Metal supports_op 限制
- [13-llama-cpp-integration.md](./13-llama-cpp-integration.md) - llama.cpp 调度参数
