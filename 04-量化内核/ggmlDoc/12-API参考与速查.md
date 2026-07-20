# 12 - API 参考与速查

## 1. Context 与张量

```c
struct ggml_init_params params = { .mem_size = 16<<20, .no_alloc = false };
struct ggml_context * ctx = ggml_init(params);

struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
ggml_set_name(t, "weight");

ggml_free(ctx);
ggml_reset(ctx);   // 重置对象链表，保留 mem_buffer
```

| API | 说明 |
|-----|------|
| `ggml_new_tensor_{1,2,3,4}d` | 创建张量 |
| `ggml_view_tensor` / `ggml_view_*` | 零拷贝 view |
| `ggml_cont` | 连续化 |
| `ggml_nbytes` / `ggml_nelements` | 大小 |
| `ggml_is_contiguous` | 连续性检查 |

## 2. 算子（LLM 常用）

```c
ggml_mul_mat(ctx, w, x);
ggml_rope(ctx, a, pos, n_dims, mode);
ggml_rms_norm(ctx, a, eps);
ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap);
ggml_get_rows(ctx, embd, tokens);
ggml_glu(ctx, gate, up, GGML_GLU_OP_SWIGLU);
ggml_mul_mat_id(ctx, experts, x, ids);
ggml_soft_max(ctx, a);
```

## 3. 建图与执行

```c
struct ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, output);

// 纯 CPU
struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads);
ggml_graph_compute(gf, &cplan);

// Backend
ggml_backend_graph_compute(backend, gf);
ggml_backend_sched_graph_compute_async(sched, gf);
ggml_backend_sched_synchronize(sched);
```

## 4. Backend / Device

```c
ggml_backend_load_all();
ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);

ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(buft, size);

ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n_backends, ...);
ggml_backend_sched_reserve(sched, worst_case_gf);
ggml_backend_sched_alloc_graph(sched, gf);
```

## 5. 内存分配

```c
ggml_gallocr_t galloc = ggml_gallocr_new_n(bufts, n_bufs);
ggml_gallocr_reserve(galloc, gf, node_ids, leaf_ids);
ggml_gallocr_alloc_graph(galloc, gf);

ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
```

## 6. GGUF

```c
struct gguf_init_params params = { .no_alloc = true, .ctx = &ctx };
struct gguf_context * meta = gguf_init_from_file("model.gguf", params);
const char * arch = gguf_get_val_str(meta, "general.architecture");
gguf_free(meta);
```

## 7. 量化

```c
quantize_row_q4_0_ref(src, dst, n);
ggml_quantize_chunk(GGML_TYPE_Q4_K, src, dst, start, n, nrows, imatrix);
ggml_quantize_requires_imatrix(GGML_TYPE_IQ2_XXS);
ggml_vec_dot_q4_0_q8_0(n, &s, src0, src1);
```

## 8. 关键源码路径

| 主题 | 路径 |
|------|------|
| 张量/图 | `src/ggml.c` |
| 内部结构 | `src/ggml-impl.h` |
| 公共 API | `include/ggml.h` |
| Backend 调度 | `src/ggml-backend.cpp` |
| 内存 | `src/ggml-alloc.c` |
| 量化 | `src/ggml-quants.c`, `src/ggml-common.h` |
| GGUF | `src/gguf.cpp` |
| CPU 执行 | `src/ggml-cpu/ggml-cpu.c` |
| CUDA | `src/ggml-cuda/ggml-cuda.cu` |
| 训练 | `src/ggml-opt.cpp` |
| 线程池 | `src/ggml-threading.cpp` |

## 9. 关键行号锚点

| 主题 | 位置 |
|------|------|
| type_traits | `ggml.c:621+` |
| build_forward | `ggml.c:6964+` |
| ggml_gallocr 结构 | `ggml-alloc.c:481-495` |
| sched split | `ggml-backend.cpp:1014+` |
| graph_compute | `ggml-cpu.c:3308` |
| CUDA mul_mat 路由 | `ggml-cuda.cu:2541+` |
| GGUF 加载 | `gguf.cpp:896-979` |

## 10. 构建命令

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON
cmake --build build -j$(nproc)
```

## 11. 环境变量

| 变量 | 用途 |
|------|------|
| `GGML_DEBUG` | 调试输出 |
| `GGML_SCHED_NO_REALLOC` | 禁止 realloc |
| `GGML_DISABLE_VULKAN` | 禁用 Vulkan |
| `GGML_CUDA_FORCE_MMQ` | CUDA 量化 matmul |
| `GGML_METAL_DEVICES` | Metal 设备 |

## 12. 文档索引

| 场景 | 文档 |
|------|------|
| 入门 | [01](./01-GGML项目总览.md) [02](./02-整体架构.md) [03](./03-张量模型与计算图.md) |
| 调度/内存 | [04](./04-Backend抽象与调度器.md) [05](./05-内存分配器.md) |
| 量化/GGUF | [06](./06-量化系统.md) [07](./07-GGUF文件格式.md) |
| Backend | [08](./08-CPU后端.md) [09](./09-GPU后端.md) [10](./10-其他后端.md) |
| 进阶 | [14](./14-ggml-opt与线程系统.md) [15](./15-Metal与Vulkan深度解析.md) |
| llama 集成 | [13](./13-与llama.cpp集成.md) |

## 上游

- https://github.com/ggml-org/ggml
- MIT License
