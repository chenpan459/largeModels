# 05 - GGML 计算引擎 (ggml/)

## 1. 模块概述

**GGML** (Georgi Gerganov Machine Learning) 是 llama.cpp 的底层张量计算库，提供：

- 张量定义与操作 (类似 micro-framework)
- 计算图 (computation graph) 构建与执行
- 多后端 (CPU/GPU/NPU) 调度
- 量化内核 (1-bit 到 FP32)
- GGUF 文件格式读写

- **版本**: 0.15.2
- **构建目标**: `libggml`
- **语言**: C/C++ (核心 C，后端 C++)

## 2. 目录结构

```
ggml/
├── include/              # 公开头文件
│   ├── ggml.h            # 核心 API
│   ├── ggml-cpu.h        # CPU 后端
│   ├── ggml-backend.h    # Backend 抽象
│   ├── ggml-opt.h        # 优化器 (训练)
│   ├── ggml-alloc.h      # 内存分配
│   ├── gguf.h            # GGUF 格式
│   └── ggml-cpp.h        # C++ wrapper
├── src/                  # 核心实现
│   ├── ggml.c            # 张量操作定义
│   ├── ggml.cpp          # C++ 扩展
│   ├── ggml-alloc.c      # Graph 内存分配器
│   ├── ggml-backend.cpp  # Backend 调度 (2,371 行)
│   ├── ggml-backend-reg.cpp  # Backend 注册
│   ├── ggml-quants.c     # 量化/反量化
│   ├── gguf.cpp          # GGUF 读写 (1,688 行)
│   ├── ggml-opt.cpp      # 优化器
│   └── ggml-*/           # 各硬件后端
└── CMakeLists.txt
```

## 3. 核心概念

### 3.1 张量 (ggml_tensor)

```c
struct ggml_tensor {
    enum ggml_type type;     // F32, F16, Q4_K, ...
    int64_t ne[GGML_MAX_DIMS];  // 各维度大小
    size_t  nb[GGML_MAX_DIMS];  // stride
    void  * data;            // 数据指针
    char    name[GGML_MAX_NAME];
    struct ggml_tensor * src[GGML_MAX_SRC];  // 依赖的源张量
    enum ggml_op op;         // 产生此张量的操作
};
```

### 3.2 计算图 (ggml_cgraph)

```c
struct ggml_cgraph {
    struct ggml_tensor ** nodes;     // 拓扑排序的节点
    int n_nodes;
    struct ggml_tensor ** leafs;     // 输入/权重 (leaf nodes)
    int n_leafs;
};
```

构建方式：每个 `ggml_*()` 操作自动添加到当前 context 的 graph 中。

### 3.3 Backend 抽象

```c
struct ggml_backend {
    const char * name;
    // 分配 buffer, 执行 graph, 同步, ...
};

struct ggml_backend_buffer_type {
    // 描述内存类型 (CPU RAM, GPU VRAM, ...)
};
```

### 3.4 Backend Scheduler

`ggml_backend_sched` 是 llama.cpp 多设备推理的核心：

- 自动将 graph 节点分配到合适的 backend
- 管理跨 backend 的数据拷贝
- 支持 pipeline 并行 (layer split across GPUs)
- 支持 tensor parallelism (row split)

## 4. 硬件后端

| 后端 | 目录 | CMake 选项 | 平台 |
|------|------|-----------|------|
| CPU | `ggml-cpu/` | `GGML_CPU=ON` (默认) | 全平台 |
| CUDA | `ggml-cuda/` | `GGML_CUDA=ON` | NVIDIA GPU |
| Metal | `ggml-metal/` | `GGML_METAL=ON` | Apple GPU |
| Vulkan | `ggml-vulkan/` | `GGML_VULKAN=ON` | 跨平台 GPU |
| SYCL | `ggml-sycl/` | `GGML_SYCL=ON` | Intel GPU |
| HIP | `ggml-hip/` | `GGML_HIP=ON` | AMD GPU |
| MUSA | `ggml-musa/` | `GGML_MUSA=ON` | 摩尔线程 GPU |
| OpenCL | `ggml-opencl/` | `GGML_OPENCL=ON` | 通用 GPU |
| OpenVINO | `ggml-openvino/` | `GGML_OPENVINO=ON` | Intel 推理 |
| CANN | `ggml-cann/` | `GGML_CANN=ON` | 华为 Ascend |
| RPC | `ggml-rpc/` | `GGML_RPC=ON` | 远程 GPU |
| WebGPU | `ggml-webgpu/` | `GGML_WEBGPU=ON` | 浏览器 |
| Hexagon | `ggml-hexagon/` | `GGML_HEXAGON=ON` | 高通 NPU |
| BLAS | `ggml-blas/` | `GGML_BLAS=ON` | BLAS 加速 |
| zDNN | `ggml-zdnn/` | `GGML_ZDNN=ON` | IBM zDNN |
| ZenDNN | `ggml-zendnn/` | `GGML_ZENDNN=ON` | AMD ZenDNN |
| VirtGPU | `ggml-virtgpu/` | - | 虚拟 GPU |

### 4.1 CPU 后端特性

- **SIMD 优化**: AVX, AVX2, AVX512, AVX-VNNI, AMX (Intel)
- **ARM**: NEON, dotprod, i8mm, SVE
- **RISC-V**: RVV, ZVFH, ZFH
- **量化内核**: 所有 Q-type 的 dot product
- **Repack**: 运行时 Q4_0 -> Q4_X 转换 (`GGML_CPU_REPACK`)
- **KleidiAI**: ARM KleidiAI 优化 (`GGML_CPU_KLEIDIAI`)
- **llamafile**: 多线程 SGEMM (`GGML_LLAMAFILE`)

### 4.2 CUDA 后端特性

- 自定义 CUDA kernel (非 cuBLAS 依赖)
- Flash Attention kernel
- 量化 matmul (MMQ - Matrix Multiplication Quantized)
- Multi-GPU: Pipeline Parallel + Tensor Parallel
- CUDA Graphs 支持 (`GGML_CUDA_GRAPHS`)
- FP16/BF16/INT8/INT4 混合精度

## 5. 量化系统

### 5.1 量化类型 (ggml_type)

| 类型 | 每权重 bit | 说明 |
|------|-----------|------|
| F32 | 32 | 全精度 |
| F16 | 16 | 半精度 |
| BF16 | 16 | Brain Float |
| Q4_0 | ~4.5 | 4-bit, block=32 |
| Q4_1 | ~5.0 | 4-bit + min |
| Q5_0/1 | ~5.5/6.0 | 5-bit |
| Q8_0 | 8.5 | 8-bit |
| Q2_K - Q6_K | 2-6 | K-quants (超块) |
| IQ1_S/M, IQ2_*, IQ3_*, IQ4_* | 1-4 | Importance 量化 |
| TQ1_0, TQ2_0 | 1-2 | Ternary 量化 |
| MXFP4 | 4 | MX 浮点格式 |

### 5.2 量化内核

`ggml-quants.c` 实现：
- `quantize_row_*()`: 浮点 -> 量化
- `dequantize_row_*()`: 量化 -> 浮点
- `ggml_vec_dot_*()`: 量化向量点积 (推理核心)

## 6. GGUF 格式

GGUF (GGML Unified Format) 是自描述模型文件格式：

```
[Header]
  magic: "GGUF"
  version: uint32
  n_tensors: uint64
  n_kv: uint64

[Metadata KV pairs]
  general.architecture = "llama"
  llama.context_length = 8192
  llama.embedding_length = 4096
  tokenizer.ggml.model = "llama"
  ...

[Tensor Info]
  name, n_dims, dims[], type, offset

[Tensor Data]
  raw weight bytes (aligned)
```

实现位于 `ggml/src/gguf.cpp` 和 Python 库 `gguf-py/`。

## 7. 主要 CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `GGML_CPU` | ON | CPU 后端 |
| `GGML_CUDA` | OFF | NVIDIA CUDA |
| `GGML_METAL` | 平台 | Apple Metal |
| `GGML_VULKAN` | OFF | Vulkan GPU |
| `GGML_NATIVE` | ON | 针对本机 CPU 优化 |
| `GGML_BACKEND_DL` | OFF | 后端动态加载 |
| `GGML_LLAMAFILE` | ON | llamafile SGEMM |
| `GGML_CUDA_GRAPHS` | ON | CUDA Graph 优化 |
| `GGML_CPU_REPACK` | ON | 运行时 weight repack |

## 8. 与 llama.cpp 的关系

```
llama.cpp (src/)
    |
    |  build_graph() 构建 ggml_cgraph
    |  使用 ggml_mul_mat, ggml_rope, ggml_norm 等操作
    |
    v
ggml_backend_sched_graph_compute()
    |
    v
ggml-cpu / ggml-cuda / ggml-metal ... 执行
```

llama.cpp 是 ggml 最大的应用，许多新 op (如 `ggml_gated_delta_net`, `ggml_flash_attn_ext`) 首先在 llama.cpp 中驱动开发，然后下沉到 ggml。

## 9. 扩展指南

| 需求 | 位置 |
|------|------|
| 新 ggml 操作 | `ggml.c` 添加 enum + forward + backward |
| 新量化类型 | `ggml-quants.c` + `ggml-common.h` |
| 新 GPU 后端 | 创建 `ggml-xxx/` 目录，实现 backend 接口 |
| 新 CPU 内核 | `ggml-cpu/` 中添加 arch-specific 实现 |
