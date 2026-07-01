# 11 - 构建系统

## 1. CMake 概览

| 文件 | 行数 | 职责 |
|------|------|------|
| `CMakeLists.txt` | ~504 | 根构建选项 |
| `src/CMakeLists.txt` | ~517 | 库目标拆分与 Backend 注册 |

版本：**GGML 0.15.3**

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
```

---

## 2. 库目标

```
ggml-base    # 核心（始终构建）
ggml         # 注册层
ggml-cpu     # CPU Backend（默认 ON）
ggml-cuda    # CUDA（可选）
ggml-metal   # Metal（macOS 默认）
ggml-vulkan  # Vulkan（可选）
...
```

### 2.1 ggml-base 源文件

```
ggml.c, ggml.cpp
ggml-alloc.c
ggml-backend.cpp, ggml-backend-meta.cpp
ggml-quants.c
gguf.cpp
ggml-opt.cpp
```

### 2.2 ggml 源文件

```
ggml-backend-reg.cpp
ggml-backend-dl.cpp  (GGML_BACKEND_DL)
```

---

## 3. 通用选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `BUILD_SHARED_LIBS` | ON | 动态库 |
| `GGML_BACKEND_DL` | OFF | Backend 插件化 |
| `GGML_NATIVE` | ON | 本机 CPU 优化 |
| `GGML_OPENMP` | ON | OpenMP 并行 |
| `GGML_LTO` | OFF | 链接时优化 |
| `GGML_ALL_WARNINGS` | ON | 编译警告 |

---

## 4. CPU 选项

| 选项 | 说明 |
|------|------|
| `GGML_CPU` | 启用 CPU Backend |
| `GGML_AVX/AVX2/AVX512` | x86 SIMD |
| `GGML_AVX512_VNNI` | int8 dot product |
| `GGML_AMX_INT8/AMX_BF16` | Intel AMX |
| `GGML_CPU_REPACK` | Q4_0 运行时 repack |
| `GGML_LLAMAFILE` | llamafile SGEMM |
| `GGML_CPU_KLEIDIAI` | ARM KleidiAI |
| `GGML_CPU_ALL_VARIANTS` | 多 ISA 变体（需 BACKEND_DL） |

---

## 5. GPU 选项

| 选项 | 说明 |
|------|------|
| `GGML_CUDA` | NVIDIA CUDA |
| `GGML_HIP` | AMD ROCm |
| `GGML_MUSA` | 摩尔线程 |
| `GGML_METAL` | Apple Metal（macOS 默认 ON） |
| `GGML_METAL_EMBED_LIBRARY` | Shader 嵌入二进制 |
| `GGML_VULKAN` | Vulkan |
| `GGML_SYCL` | Intel SYCL |
| `GGML_CUDA_FA` | Flash Attention CUDA |
| `GGML_CUDA_GRAPHS` | CUDA Graph |
| `GGML_CUDA_FORCE_MMQ` | 强制 MMQ |
| `GGML_CUDA_FORCE_CUBLAS` | 强制 cuBLAS |

---

## 6. 其他 Backend 选项

| 选项 | 说明 |
|------|------|
| `GGML_BLAS` | BLAS 加速 |
| `GGML_RPC` | 远程 GPU |
| `GGML_OPENCL` | OpenCL |
| `GGML_HEXAGON` | 高通 NPU |
| `GGML_WEBGPU` | WebGPU |
| `GGML_CANN` | 华为 Ascend |
| `GGML_OPENVINO` | Intel OpenVINO |
| `GGML_ZENDNN` | AMD ZenDNN |
| `GGML_ZDNN` | IBM zDNN |

---

## 7. llama.cpp 中的构建

llama.cpp 根 `CMakeLists.txt` 添加 ggml 为子目录：

```cmake
add_subdirectory(ggml)
target_link_libraries(llama ggml)
```

常用组合：

```bash
# NVIDIA GPU
cmake -B build -DGGML_CUDA=ON

# Apple Silicon
cmake -B build -DGGML_METAL=ON

# CPU only
cmake -B build

# Vulkan 跨平台
cmake -B build -DGGML_VULKAN=ON
```

---

## 8. 调试选项

| 环境变量 | 效果 |
|----------|------|
| `GGML_DEBUG` | 打印图/张量 debug 信息 |
| `GGML_SCHED_NO_REALLOC` | 禁止 sched realloc |
| `GGML_DISABLE_VULKAN` | 禁用 Vulkan |
| `GGML_CUDA_FORCE_MMQ` | 强制 CUDA MMQ |

---

## 9. 相关文档

- [08-backend-cpu.md](./08-backend-cpu.md) - CPU 构建选项详解
- [09-backend-gpu.md](./09-backend-gpu.md) - GPU 构建选项详解
- `03-推理部署/llama.cppDoc/10-build-system.md` - llama.cpp 构建
