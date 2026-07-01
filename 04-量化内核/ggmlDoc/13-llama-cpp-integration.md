# 13 - 与 llama.cpp 的集成

## 1. 依赖关系

```
llama.cpp (libllama)
    |
    +-- links ggml-base, ggml, ggml-{cpu,cuda,...}
    |
    v
ggml (本仓库)
```

llama.cpp 内嵌 ggml 副本：`03-推理部署/llama.cpp/ggml/`  
独立 ggml 仓库：`04-量化内核/ggml/`

---

## 2. 调用链对照

| llama.cpp 阶段 | ggml 组件 | 源码位置 |
|----------------|-----------|----------|
| 加载 GGUF | `gguf_init_from_file` | `gguf.cpp` |
| 识别架构 | `gguf_get_val_str("general.architecture")` | `llama-arch.cpp` |
| 创建 tensor | `ggml_new_tensor` + `create_tensor` | `llama-model-loader.cpp` |
| 分配权重 buffer | `ggml_backend_alloc_ctx_tensors_from_buft` | `llama-model.cpp` |
| 创建 sched | `ggml_backend_sched_new` | `llama-context.cpp` |
| 建图 | `ggml_mul_mat`, `ggml_rope`, ... | `llama-graph.cpp`, `models/` |
| 预留 buffer | `ggml_gallocr_reserve` + `sched_reserve` | `llama-context.cpp` |
| 推理 | `ggml_backend_sched_graph_compute_async` | `llama-context.cpp` |
| 量化模型 | `quantize_q4_K` 等 | `llama-quantize` -> `ggml-quants.c` |

---

## 3. 模型加载流程

```
GGUF file
    |
    v
gguf_init_from_file(no_alloc=true)          # gguf.cpp
    |
    v
llama_model_loader
    |  parse KV -> llama_hparams
    |  create_tensor() for each weight
    |  ggml_backend_buffer for GPU/CPU
    |
    v
ggml_backend_sched_new(devices)             # 多 GPU 调度
    |
    v
llama_model ready
```

详见：`03-推理部署/llama.cppDoc/14-model-loader-deep-dive.md`

---

## 4. Decode 推理流程

```
llama_decode(batch)
    |
    v
build_graph(ubatch)                       # llama-graph.cpp
    |  ggml_mul_mat, ggml_rope, ggml_flash_attn_ext, ...
    |  -> ggml_cgraph
    |
    v
can_reuse? -> sched_alloc_graph / skip
    |
    v
ggml_backend_sched_graph_compute_async    # ggml-backend.cpp
    |  split_graph -> CUDA/CPU splits
    |  gallocr alloc/free 中间张量
    |
    v
logits in output buffer
```

详见：`03-推理部署/llama.cppDoc/15-decode-graph-reuse.md`

---

## 5. 参数映射

| llama.cpp 参数 | ggml 影响 |
|----------------|-----------|
| `n_gpu_layers` | 前 N 层权重 -> GPU buffer |
| `n_threads` | `ggml_graph_plan.n_threads` |
| `flash_attn_type` | 使用 `ggml_flash_attn_ext` vs 标准 attention |
| `type_k`, `type_v` | KV cache 张量类型 |
| `tensor_split` | Meta Backend 张量并行 |
| `pipeline_parallel` | sched `n_copies` pipeline |

---

## 6. 量化工具链

```
HF model
    |
    v
convert_hf_to_gguf.py -> GGUF (F16/F32)
    |
    v
llama-imatrix -> importance matrix
    |
    v
llama-quantize -> GGUF (Q4_K_M, ...)
    |  uses ggml-quants.c quantize_*
    |
    v
llama-server / llama-cli 加载推理
```

---

## 7. 新算子添加流程（跨仓库）

```
1. ggml.h:     添加 GGML_OP_XXX enum
2. ggml.c:     添加 ggml_xxx() 工厂函数
3. ggml-cpu/ops.cpp:  CPU 实现
4. ggml-cuda/:       CUDA kernel（可选）
5. ggml-metal/:      Metal shader（可选）
6. llama-graph.cpp:  在 build_graph 中调用新 op
7. tests/:           添加 test-backend-ops
```

---

## 8. 对照阅读路径

| 主题 | ggmlDoc | llama.cppDoc |
|------|---------|--------------|
| 架构 | 02-architecture | 02-architecture |
| 张量/图 | 03-tensor-graph | 03-src-core |
| Backend | 04-backend-scheduler | 05-ggml (概览) |
| 内存 | 05-memory-alloc | 15-decode-graph-reuse |
| 量化 | 06-quantization | 09-conversion-gguf |
| GGUF | 07-gguf-format | 14-model-loader |
| GPU | 09-backend-gpu | 00编译方法 |

---

## 9. 性能调优协同

| 目标 | ggml 侧 | llama.cpp 侧 |
|------|---------|--------------|
| GPU 加速 | `-DGGML_CUDA=ON` | `-ngl 99` |
| Flash Attn | `GGML_CUDA_FA` | `-fa on` |
| 量化 matmul | `GGML_CPU_REPACK` | 使用 Q4_K 模型 |
| 多 GPU | Meta Backend | `--tensor-split` |
| 减少延迟 | CUDA Graphs | 图复用 `can_reuse` |
| 内存 | gallocr in-place | `-c` ctx size, KV 量化 |

---

## 10. 相关链接

- ggml 上游: https://github.com/ggml-org/ggml
- llama.cpp 上游: https://github.com/ggml-org/llama.cpp
- llama.cppDoc: `03-推理部署/llama.cppDoc/`
