# 11 - 构建系统

## 1. CMake 概览

| 文件 | 行数 | 职责 |
|------|------|------|
| `CMakeLists.txt` | ~504 | 根选项、版本、Sanitizer |
| `src/CMakeLists.txt` | ~517 | 库目标、Backend 注册 |
| `cmake/common.cmake` | — | 架构检测、编译器标志 |

版本：**GGML 0.15.3**（`CMakeLists.txt` L6-9）

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
```

---

## 2. 库目标与源文件

### ggml-base（始终构建）

```
ggml.c, ggml.cpp
ggml-alloc.c
ggml-backend.cpp, ggml-backend-meta.cpp
ggml-quants.c
gguf.cpp
ggml-opt.cpp
ggml-threading.cpp
```

### ggml（注册层）

```
ggml-backend-reg.cpp
ggml-backend-dl.cpp   # GGML_BACKEND_DL
```

### Backend 库（条件）

`ggml_add_backend(CPU/CUDA/METAL/...)` 宏注册；`GGML_USE_${BACKEND}` 编译定义注入。

`ggml_add_backend_library`：DL 模式编译为 MODULE 插件。

`ggml_add_cpu_backend_variant(tag)`：多 ISA 变体（x86 15+、ARM 10+）。

---

## 3. 通用选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_SHARED_LIBS` | ON | 动态库 |
| `GGML_BACKEND_DL` | OFF | Backend 插件化 |
| `GGML_NATIVE` | ON | 本机 CPU `-march=native` |
| `GGML_OPENMP` | ON | OpenMP |
| `GGML_LTO` | OFF | 链接时优化 |
| `GGML_ALL_WARNINGS` | ON | -Wall 等 |
| `GGML_SCHED_MAX_COPIES` | **4** | Pipeline 副本数（CACHE STRING） |
| `GGML_SCHED_NO_REALLOC` | OFF | 禁止 sched realloc |

Sanitizer：`GGML_SANITIZE_THREAD/ADDRESS/UNDEFINED`

---

## 4. CPU ISA 选项（节选）

| 选项 | 说明 |
|------|------|
| `GGML_CPU` | CPU Backend ON |
| `GGML_AVX/AVX2/AVX512` | x86 SIMD |
| `GGML_AVX512_VNNI/AVX512_BF16` | int8 dot / BF16 |
| `GGML_AMX_TILE/AMX_INT8/AMX_BF16` | Intel AMX |
| `GGML_RVV/RV_ZFH/RV_ZVFH` | RISC-V |
| `GGML_CPU_REPACK` | Q4 repack buffer（默认 ON） |
| `GGML_CPU_KLEIDIAI` | ARM KleidiAI |
| `GGML_CPU_ALL_VARIANTS` | 多 ISA .so 变体 |
| `GGML_LLAMAFILE` | llamafile SGEMM |
| `GGML_BLAS` | BLAS F32 GEMM |
| `GGML_CPU_HBM` | 高带宽内存 |

PowerPC / s390x / LoongArch 各有对应 CMake 选项（见 `src/CMakeLists.txt`）。

---

## 5. GPU 选项

| 选项 | 说明 |
|------|------|
| `GGML_CUDA` | NVIDIA |
| `GGML_HIP` | AMD（共用 ggml-cuda） |
| `GGML_MUSA` | 摩尔线程 |
| `GGML_METAL` | Apple（macOS 默认 ON） |
| `GGML_METAL_EMBED_LIBRARY` | Shader 嵌入 |
| `GGML_VULKAN` | Vulkan |
| `GGML_VULKAN_CHECK_RESULTS/RUN_TESTS` | 调试 |
| `GGML_SYCL` | Intel SYCL |
| `GGML_WEBGPU` | WebGPU |
| `GGML_CUDA_FA` | Flash Attention |
| `GGML_CUDA_FA_ALL_QUANTS` | 全量化 FA |
| `GGML_CUDA_GRAPHS` | CUDA Graph（llama only） |
| `GGML_CUDA_FORCE_MMQ/CUBLAS` | 强制 kernel |
| `GGML_CUDA_NCCL` | 多卡 NCCL |

---

## 6. 其他 Backend

`GGML_RPC`, `GGML_OPENCL`, `GGML_HEXAGON`, `GGML_CANN`, `GGML_OPENVINO`, `GGML_ZENDNN`, `GGML_ZDNN`, `GGML_VIRTGPU`, `GGML_VIRTGPU_BACKEND`

---

## 7. 编译定义注入

- `GGML_USE_CUDA` 等 → `ggml` 目标
- `GGML_SCHED_MAX_COPIES` → `ggml-base`
- HIP/MUSA：定义 `GGML_USE_HIP` 等但编译 `ggml-cuda` 源

---

## 8. llama.cpp 构建

```cmake
add_subdirectory(ggml)
target_link_libraries(llama PRIVATE ggml ...)
```

常用：

```bash
# NVIDIA
cmake -B build -DGGML_CUDA=ON

# Apple
cmake -B build -DGGML_METAL=ON

# CPU only
cmake -B build

# Vulkan 跨平台
cmake -B build -DGGML_VULKAN=ON

# 插件化 Backend
cmake -B build -DGGML_BACKEND_DL=ON -DGGML_CUDA=ON
```

---

## 9. 环境变量（运行时）

| 变量 | 效果 |
|------|------|
| `GGML_DEBUG` | 图/张量 debug |
| `GGML_SCHED_NO_REALLOC` | 禁止 realloc |
| `GGML_SCHED_DEBUG_REALLOC` | realloc 追踪 |
| `GGML_DISABLE_VULKAN` | 禁用 Vulkan |
| `GGML_CUDA_FORCE_MMQ` | 强制 MMQ |
| `GGML_METAL_DEVICES` | Metal 设备选择 |
| `GGML_VULKAN_DEBUG` | Vulkan 日志 |

---

## 相关文档

- [08-backend-cpu.md](./08-backend-cpu.md)
- [09-backend-gpu.md](./09-backend-gpu.md)
- [10-other-backends.md](./10-other-backends.md)
- `03-推理部署/llama.cppDoc/10-build-system.md`
