# 12 - 公共 C API 概览

## 1. 头文件清单

| 头文件 | 行数(约) | 用途 |
|--------|----------|------|
| `ggml.h` | 2,863 | 核心：张量、算子、图 |
| `ggml-backend.h` | 435 | Backend、调度器 |
| `ggml-alloc.h` | 86 | 图级分配器 |
| `gguf.h` | 210 | GGUF 格式 |
| `ggml-cpu.h` | 151 | CPU 图计算 |
| `ggml-cuda.h` | 50 | CUDA 初始化 |
| `ggml-metal.h` | 61 | Metal |
| `ggml-vulkan.h` | 29 | Vulkan |
| `ggml-opt.h` | 256 | 优化器（AdamW/SGD） |
| `ggml-cpp.h` | 39 | C++ 辅助 |

---

## 2. 核心 API 模块

### 2.1 初始化与张量

```c
struct ggml_context * ggml_init(struct ggml_init_params params);
void ggml_free(struct ggml_context * ctx);

struct ggml_tensor * ggml_new_tensor_1d(ctx, type, ne0);
struct ggml_tensor * ggml_new_tensor_2d(ctx, type, ne0, ne1);
struct ggml_tensor * ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
struct ggml_tensor * ggml_view_tensor(ctx, src);
```

### 2.2 常用算子

```c
struct ggml_tensor * ggml_mul_mat(ctx, a, b);
struct ggml_tensor * ggml_add(ctx, a, b);
struct ggml_tensor * ggml_rope(ctx, a, pos, n_dims, mode);
struct ggml_tensor * ggml_rms_norm(ctx, a, eps);
struct ggml_tensor * ggml_soft_max(ctx, a);
struct ggml_tensor * ggml_flash_attn_ext(ctx, q, k, v, mask, ...);
struct ggml_tensor * ggml_get_rows(ctx, embd, tokens);
struct ggml_tensor * ggml_glu(ctx, a, b, type);
struct ggml_tensor * ggml_mul_mat_id(ctx, as, b, ids);
```

### 2.3 图构建与执行

```c
struct ggml_cgraph * ggml_new_graph(ctx);
void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
```

### 2.4 Backend

```c
ggml_backend_t ggml_backend_init_best(void);
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
enum ggml_status ggml_backend_graph_compute(backend, cgraph);

ggml_backend_sched_t ggml_backend_sched_new(backends, buft, n_backends, ...);
bool ggml_backend_sched_graph_compute(sched, cgraph);
bool ggml_backend_sched_graph_compute_async(sched, cgraph);
void ggml_backend_sched_synchronize(sched);
```

### 2.5 GGUF

```c
struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
const void * gguf_get_tensor_data(const struct gguf_context * ctx, const char * name);
const char * gguf_get_val_str(const struct gguf_context * ctx, const char * key);
```

### 2.6 量化

```c
size_t ggml_quantize_chunk(type, src, dst, start, nrows, n_per_row, imatrix);
void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k);
```

---

## 3. 关键枚举

### ggml_type（部分）

```
GGML_TYPE_F32=0, F16=1, Q4_0=2, Q4_1=3, ...
Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15, ...
IQ4_NL=25, MXFP4=38, ...
```

### ggml_status

```
GGML_STATUS_ALLOC_FAILED = -2
GGML_STATUS_FAILED = -1
GGML_STATUS_SUCCESS = 0
GGML_STATUS_ABORTED = 1
```

---

## 4. 最小示例

```c
#include "ggml.h"
#include "ggml-cpu.h"

int main() {
    struct ggml_init_params params = { .mem_size = 16*1024*1024 };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    struct ggml_tensor * c = ggml_mul_mat(ctx, a, b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    struct ggml_cplan cplan = ggml_graph_plan(gf, 4);
    ggml_graph_compute(gf, &cplan);

    ggml_free(ctx);
    return 0;
}
```

---

## 5. 相关文档

- [03-tensor-graph.md](./03-tensor-graph.md) - 张量与图详解
- [04-backend-scheduler.md](./04-backend-scheduler.md) - Backend API 详解
- [07-gguf-format.md](./07-gguf-format.md) - GGUF API
