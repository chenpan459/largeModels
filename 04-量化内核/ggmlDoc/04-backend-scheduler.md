# 04 - Backend 抽象与调度器

## 1. 模块概述

| 文件 | 行数 | 职责 |
|------|------|------|
| `include/ggml-backend.h` | ~435 | 公共 Backend API |
| `src/ggml-backend.cpp` | ~2,371 | Backend 抽象、调度器 |
| `src/ggml-backend-reg.cpp` | ~586 | Backend 注册表 |
| `src/ggml-backend-meta.cpp` | ~2,263 | 张量并行 Meta Backend |
| `src/ggml-backend-dl.cpp` | — | 动态加载 `.so` |
| `src/ggml-backend-impl.h` | — | 五层 vtable（API v2） |

## 2. 五层接口

```
ggml_backend_buffer_type_i
  get_name, get_alignment, get_max_size, get_alloc_size, is_host, alloc_buffer

ggml_backend_buffer_i
  get_base, init_tensor, set/get_tensor, cpy_tensor, clear, reset

ggml_backend_i
  get_name, graph_compute, synchronize,
  set/get_tensor_async, cpy_tensor_async, event_* 

ggml_backend_device_i
  get_name, get_type, get_memory, supports_op, supports_buft,
  offload_op, buffer_from_host_ptr, get_host_buffer_type

ggml_backend_reg_i
  get_name, get_device_count, get_device, get_proc_address
```

### 关键类型

```c
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct ggml_backend_buffer     * ggml_backend_buffer_t;
typedef struct ggml_backend            * ggml_backend_t;
typedef struct ggml_backend_device     * ggml_backend_dev_t;
typedef struct ggml_backend_reg        * ggml_backend_reg_t;
typedef struct ggml_backend_sched      * ggml_backend_sched_t;
```

## 3. Backend 注册表

### 静态注册顺序（`ggml-backend-reg.cpp` L115-166）

```
CUDA → Metal → SYCL → Vulkan → WebGPU → zDNN → VirtGPU → OpenCL
→ ZenDNN → Hexagon → CANN → BLAS → RPC → OpenVINO → CPU（最后）
```

**CPU 最后注册** = sched 中最低优先级兜底。

运行时：`GGML_DISABLE_VULKAN=1` 可禁用 Vulkan。

### 动态加载（`GGML_BACKEND_DL`）

```c
ggml_backend_load("libggml-cuda.so");
ggml_backend_load_all();              // 扫描可执行文件目录
ggml_backend_init_by_name("CUDA", NULL);
ggml_backend_init_best();             // 优先 GPU，否则 CPU
```

### 设备枚举

```c
ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
int n = ggml_backend_reg_dev_count(reg);
ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, gpu_id);
ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
```

## 4. 调度器 `ggml_backend_sched`

### 结构（`ggml-backend.cpp` L774-828）

| 字段 | 说明 |
|------|------|
| `backends[]`, `bufts[]` | Backend 列表与 buffer type |
| `galloc` | 图级分配器 |
| `hash_set` + `hv_tensor_backend_ids` | 张量 → Backend ID |
| `hv_tensor_copies` | 跨 Backend pipeline 副本 |
| `splits[]`, `n_splits` | 图切分片段 |
| `n_copies`, `cur_copy`, `events[][]` | Pipeline（默认 4 副本） |
| `graph` | 插入 copy 节点后的修改版图 |
| `ctx`, `context_buffer` | sched 内部 context（`no_alloc=true`） |
| `callback_eval` | 逐节点评估回调（调试） |
| `op_offload` | GPU offload 小 batch 开关 |
| `debug_realloc` | realloc 追踪 |

### 关键 API

| 函数 | 行号(约) | 职责 |
|------|----------|------|
| `ggml_backend_sched_new` | L1727 | 创建调度器 |
| `ggml_backend_sched_split_graph` | L1014 | 3-pass 分配 + 切分 |
| `ggml_backend_sched_compute_splits` | L1541 | copy + 执行 |
| `ggml_backend_sched_graph_compute` | — | 同步入口 |
| `ggml_backend_sched_graph_compute_async` | L1889 | **llama 使用** |
| `ggml_backend_sched_reserve` | — | worst-case 预分配 |
| `ggml_backend_sched_alloc_graph` | — | 当前图分配 |
| `ggml_backend_sched_reset` | — | 新图前重置 |
| `ggml_backend_sched_synchronize` | — | 等待 async 完成 |

## 5. 三 Pass 调度算法

```
Pass 1: 已分配 buffer 的张量绑定 Backend
        WEIGHTS buffer on GPU → GPU Backend
        INPUT flag → 通常 sched 最后一个 Backend

Pass 2: 从已分配节点向相邻扩展同一 Backend
        跳过 CPU（最低优先级）除非权重在 CPU

Pass 3: 未分配节点
        supports_op(dev, tensor) && supports_buft(dev, buft)
        失败 → 切 split + 跨设备 copy
```

辅助函数：

- `ggml_backend_sched_backend_id_from_cur()`（L878）
- `ggml_backend_sched_backend_from_buffer()`（L845）

## 6. 图切分与 Pipeline

```
完整 cgraph
  → split_graph → splits[0..n_splits-1]
      split 0: Backend A (nodes 0..k)
      [copy tensor X: A → B]
      split 1: Backend B (nodes k+1..m)
  → compute_splits (n_copies 个 buffer 副本轮转)
  → synchronize
```

`GGML_SCHED_MAX_COPIES`（默认 4）：CMake CACHE 变量，编译进 `ggml-base`。

## 7. 设备能力查询

```c
bool ggml_backend_dev_supports_op(dev, op_tensor);
bool ggml_backend_dev_supports_buft(dev, buft);
bool ggml_backend_dev_offload_op(dev, op_tensor);  // Metal 小 batch 阈值
```

### supports_op 典型限制（CUDA）

- tensor 必须在对应 GPU 的 CUDA buffer
- split buffer 模式**仅** `MUL_MAT` / `MUL_MAT_ID`
- 许多 op 要求 `ggml_is_contiguous`
- UNARY/GLU 按子类型白名单

CPU：**支持全部 op**（兜底）。

## 8. Async 与 Event API

```c
ggml_backend_set_tensor_async(backend, tensor, data, offset, size);
ggml_backend_get_tensor_async(backend, tensor, data, offset, size);
ggml_backend_cpy_tensor_async(dst_backend, src_backend, tensor);

ggml_backend_event_t event = ggml_backend_event_new(backend);
ggml_backend_event_record(event, backend);
ggml_backend_event_wait(backend, event);
```

用于 H2D/D2H 与 compute 重叠。

## 9. Meta Backend（张量并行）

`ggml-backend-meta.cpp`（~2,263 行）：

- 包装多个 simple 设备
- `ggml_backend_meta_split_state`：按 axis 切分 Q/K/V
- llama.cpp 多 GPU `--tensor-split` / `--split-mode layer` 相关路径
- `ggml_backend_meta_alloc_ctx_tensors_from_buft()` 静态去重分配

## 10. llama.cpp 使用流程

```
模型加载:
  ggml_backend_reg → 选 devices
  ggml_backend_sched_new(backends, ...)
  ggml_backend_alloc_ctx_tensors_from_buft(ctx, gpu_buft)

每次 decode:
  ggml_backend_sched_reset(sched)
  build_graph → ggml_cgraph
  ggml_backend_sched_alloc_graph(sched, gf)
  ggml_backend_sched_graph_compute_async(sched, gf)
  ggml_backend_sched_synchronize(sched)   # pipeline 复用前必须
```

初始化时 `graph_reserve(worst_case)` → `sched_reserve` + `gallocr_reserve`。

## 11. MoE 拷贝优化

`compute_splits`（L1576+）：`MUL_MAT_ID` 只拷贝用到的 expert slice，非整权重 tensor。

## 12. 调试环境变量

| 变量 | 作用 |
|------|------|
| `GGML_SCHED_NO_REALLOC` | 禁止 gallocr realloc |
| `GGML_SCHED_DEBUG_REALLOC` | 打印 realloc 追踪 |
| `GGML_DISABLE_VULKAN` | 运行时禁用 Vulkan |

## 相关文档

- [05-memory-alloc.md](./05-memory-alloc.md)
- [15-metal-vulkan-deep.md](./15-metal-vulkan-deep.md)
- [13-llama-cpp-integration.md](./13-llama-cpp-integration.md)
