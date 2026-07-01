# 01 - 项目总览

## 1. 项目简介

**GGML** (Georgi Gerganov Machine Learning) 是一个用 C/C++ 实现的轻量级张量计算库，是 llama.cpp 的底层计算引擎。它提供：

- 张量定义与 88+ 种算子（`ggml_op`）
- 静态计算图构建与执行
- 42+ 种数据类型（含大量量化格式）
- 多硬件 Backend 抽象与自动调度
- GGUF 模型文件格式读写

### 核心设计哲学

- **最小依赖**：核心为纯 C，无 Python/PyTorch 运行时
- **惰性建图**：算子 API 只构建图，显式调用才计算
- **预分配内存**：Context 使用 bump allocator，避免运行时 malloc
- **Backend 可插拔**：CPU/CUDA/Metal/Vulkan 等统一接口

## 2. 版本与规模

| 指标 | 数值 |
|------|------|
| 版本 | **0.15.3** (`CMakeLists.txt` L6-8) |
| 核心 `ggml.c` | ~7,815 行 |
| 公共头文件 `ggml.h` | ~2,863 行 |
| 量化 `ggml-quants.c` | ~5,591 行 |
| Backend 调度 `ggml-backend.cpp` | ~2,371 行 |
| CPU 算子 `ggml-cpu/ops.cpp` | ~11,514 行 |
| CUDA 主文件 `ggml-cuda.cu` | ~5,721 行 |
| Vulkan `ggml-vulkan.cpp` | ~18,696 行 |
| Metal Shader `ggml-metal.metal` | ~10,754 行 |

## 3. 构建产物

| 库 | 说明 |
|----|------|
| `ggml-base` | 核心：张量、图、量化、GGUF、Backend 接口、调度器 |
| `ggml` | 注册层：Backend 发现、动态加载 |
| `ggml-cpu` | CPU Backend |
| `ggml-cuda` / `ggml-metal` / `ggml-vulkan` | GPU Backend（可选） |

```
ggml-base  ← ggml.c, ggml-backend.cpp, ggml-quants.c, gguf.cpp, ggml-alloc.c
ggml       ← ggml-backend-reg.cpp, ggml-backend-dl.cpp
ggml-*     ← 各硬件 Backend 独立库
```

## 4. 目录结构

```
ggml/
├── include/              # 21 个公共头文件
│   ├── ggml.h            # 核心 API
│   ├── ggml-backend.h    # Backend / 调度器
│   ├── ggml-alloc.h      # 图级分配器
│   ├── gguf.h            # GGUF 格式
│   └── ggml-{cpu,cuda,metal,vulkan,...}.h
├── src/
│   ├── ggml.c            # 张量 + 算子 + 图
│   ├── ggml-backend.cpp  # Backend 抽象 + sched
│   ├── ggml-alloc.c      # 内存分配
│   ├── ggml-quants.c     # 量化参考实现
│   ├── gguf.cpp          # GGUF 读写
│   ├── ggml-cpu/         # CPU Backend
│   ├── ggml-cuda/        # CUDA (~80 .cu 文件)
│   ├── ggml-metal/       # Apple Metal
│   ├── ggml-vulkan/      # Vulkan
│   └── ggml-{sycl,hip,rpc,...}/
├── examples/             # gpt-2, mnist, simple 等
├── tests/                # 算子回归测试
├── docs/                 # 官方文档
└── CMakeLists.txt
```

## 5. 核心特性

### 5.1 数据类型

42 种 `ggml_type`，包括：

- 浮点：F32, F16, BF16
- 标准量化：Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- K-quants：Q2_K ~ Q8_K（超块 256 元素）
- Importance：IQ1_S/M, IQ2_XXS, IQ3_XXS, IQ4_NL 等
- 新格式：MXFP4, NVFP4, TQ1_0, TQ2_0

### 5.2 算子

88 种 `ggml_op`，LLM 相关重点：

| 算子 | 用途 |
|------|------|
| `MUL_MAT` | 矩阵乘（Attention/FFN 核心） |
| `MUL_MAT_ID` | MoE 专家路由矩阵乘 |
| `ROPE` | 旋转位置编码 |
| `RMS_NORM` | RMSNorm |
| `SOFT_MAX` | Softmax |
| `FLASH_ATTN_EXT` | Flash Attention |
| `GET_ROWS` | Embedding lookup |
| `GLU` | SwiGLU/GeGLU |
| `RWKV_WKV6/7` | RWKV 专用 |

### 5.3 硬件 Backend

| Backend | 平台 |
|---------|------|
| CPU | x86 AVX/AMX, ARM NEON/SVE, RISC-V RVV |
| CUDA | NVIDIA GPU |
| Metal | Apple Silicon |
| Vulkan | 跨平台 GPU |
| SYCL | Intel/跨厂商 |
| HIP | AMD GPU |
| RPC | 远程 GPU |
| WebGPU | 浏览器 |
| Hexagon | 高通 NPU |
| CANN | 华为 Ascend |

## 6. 典型使用场景

| 场景 | 入口 |
|------|------|
| LLM 推理 | llama.cpp（最大用户） |
| 语音 | whisper.cpp |
| 模型量化 | llama-quantize + `quantize_*` |
| 学习张量库 | `examples/simple/` |
| 自定义算子 | `ggml.c` 添加 op + Backend 实现 |

## 7. 与 llama.cpp 关系

GGML 是 **计算层**，llama.cpp 是 **应用层**：

```
llama.cpp: 模型架构、KV cache、采样、API
    ↓ 调用 ggml_* 算子建图
ggml: 张量、图、Backend 调度、量化 kernel
    ↓
CPU/GPU 硬件
```

详见 [13-llama-cpp-integration.md](./13-llama-cpp-integration.md)。
