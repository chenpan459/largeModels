# 01 - 项目总览

## 1. 项目简介

**GGML** (Georgi Gerganov Machine Learning) 是用 C/C++ 实现的轻量级张量计算库，是 **llama.cpp、whisper.cpp** 等项目的底层计算引擎。

核心能力：

- 张量定义与 **88+ 种算子**（`ggml_op`）
- **惰性建图**：算子 API 只构建计算图，显式调用才执行
- **42+ 种数据类型**（含大量量化 block 格式）
- **多硬件 Backend** 抽象与自动调度（CPU/CUDA/Metal/Vulkan/SYCL/…）
- **GGUF** 自描述模型文件格式读写
- **图级内存分配**（in-place 复用，峰值远小于节点总和）
- **ggml-opt** 训练/优化器路径（AdamW/SGD）

### 核心设计哲学

| 原则 | 实现 |
|------|------|
| 最小依赖 | 核心纯 C，无 Python/PyTorch 运行时 |
| 预分配内存 | Context bump allocator；图级 gallocr |
| Backend 可插拔 | 统一 `ggml_backend_i` vtable |
| 量化优先 | 权重 Q4/Q8 + 激活 Q8 点积是推理主路径 |
| 向后兼容 | 新 `ggml_type` 只在 enum 末尾追加 |

## 2. 版本与规模

| 指标 | 数值 |
|------|------|
| 版本 | **0.15.3**（`CMakeLists.txt` L6-9） |
| 核心 `ggml.c` | ~7,815 行 |
| 公共头 `ggml.h` | ~2,863 行 |
| 量化 `ggml-quants.c` | ~5,591 行 |
| Backend 调度 `ggml-backend.cpp` | ~2,371 行 |
| 内存分配 `ggml-alloc.c` | ~1,248 行 |
| CPU 算子 `ggml-cpu/ops.cpp` | ~11,514 行 |
| CUDA 主文件 `ggml-cuda.cu` | ~5,721 行（+65 个 .cu） |
| Vulkan `ggml-vulkan.cpp` | ~18,696 行 |
| Metal Shader `ggml-metal.metal` | ~10,754 行 |
| GGUF `gguf.cpp` | ~1,688 行 |
| 训练 `ggml-opt.cpp` | 优化器与损失算子 |

## 3. 构建产物（库拆分）

```
ggml-base   ← 核心：张量、图、量化、GGUF、Backend 接口、调度器、Meta、opt、threading
ggml        ← 注册层：Backend 发现、动态加载（ggml-backend-reg.cpp）
ggml-cpu    ← CPU Backend（默认 ON）
ggml-cuda   ← NVIDIA / HIP / MUSA（共用源码树）
ggml-metal  ← Apple（macOS 默认 ON）
ggml-vulkan ← 跨平台 GPU
ggml-{sycl,hip,rpc,webgpu,virtgpu,...}  ← 条件编译
```

链接关系：

```
llama.cpp (libllama)
  → ggml-base + ggml + ggml-{cpu,cuda,...}
```

## 4. 完整目录结构

```
ggml/
├── include/                    # 22 个公共头文件
│   ├── ggml.h                  # 核心 API、enum、struct
│   ├── ggml-backend.h          # Backend / sched / device
│   ├── ggml-alloc.h            # 图级分配器
│   ├── gguf.h                  # GGUF 格式
│   └── ggml-{cpu,cuda,metal,vulkan,sycl,...}.h
├── src/
│   ├── ggml.c                  # 张量 + 算子工厂 + 图构建
│   ├── ggml.cpp                # C++ 异常/backtrace 钩子（26 行）
│   ├── ggml-impl.h             # 内部：context、cgraph、hash_set
│   ├── ggml-backend.cpp        # Backend 抽象 + sched
│   ├── ggml-backend-reg.cpp    # Backend 注册表
│   ├── ggml-backend-meta.cpp   # 张量并行 Meta Backend
│   ├── ggml-backend-dl.cpp     # 动态加载 .so 插件
│   ├── ggml-backend-impl.h     # 五层 vtable（API version 2）
│   ├── ggml-alloc.c            # tallocr / dyn_tallocr / gallocr
│   ├── ggml-quants.c           # 量化参考实现
│   ├── ggml-common.h           # 量化 block 结构（跨 backend 共享）
│   ├── ggml-quants.h           # quant/dequant/vec_dot 声明
│   ├── ggml-opt.cpp            # 训练优化器
│   ├── ggml-threading.cpp      # 线程池
│   ├── gguf.cpp                # GGUF 读写
│   ├── ggml-cpu/               # CPU Backend + arch/*/ + amx/ + kleidiai/
│   ├── ggml-cuda/              # 65 个 .cu 文件
│   ├── ggml-metal/             # 多文件 Metal 架构
│   ├── ggml-vulkan/            # vulkan-shaders/ + shader-gen
│   └── ggml-{sycl,hip,rpc,webgpu,virtgpu,hexagon,cann,...}/
├── cmake/                      # common.cmake、arch 检测
├── examples/                   # simple, gpt-2, mnist, magika...
├── tests/                      # test-backend-ops.cpp 等
├── docs/                       # 官方 gguf.md
└── CMakeLists.txt
```

## 5. 数据类型（42 种 `ggml_type`）

| 类别 | 类型 | 说明 |
|------|------|------|
| 浮点 | F32, F16, BF16 | 标准浮点 |
| 1-bit | **Q1_0**, IQ1_S/M, TQ1_0 | 极低 bit |
| 2-bit | Q2_K, IQ2_XXS/XS/S, TQ2_0 | K-quants / Importance |
| 3-bit | Q3_K_* , IQ3_* | |
| 4-bit | Q4_0, Q4_1, Q4_K_*, IQ4_* | 最常用推理格式 |
| 5-bit | Q5_0, Q5_1, Q5_K_* | |
| 8-bit | Q8_0, Q8_1, Q8_K | 激活量化常用 Q8_0 |
| FP4 | MXFP4, NVFP4 | OCP / NVIDIA 新格式 |

**兼容性**：新类型只在 enum 末尾追加；已移除类型（Q4_2 等）保留 enum 位并注释 "removed"。

## 6. 算子（88 种 `ggml_op`）

LLM 推理重点：

| 算子 | 用途 |
|------|------|
| `MUL_MAT` | 矩阵乘（Attention QK^T、FFN 核心） |
| `MUL_MAT_ID` | MoE 专家路由矩阵乘 |
| `ROPE` | 旋转位置编码 |
| `RMS_NORM` | RMSNorm |
| `SOFT_MAX` | Softmax |
| `FLASH_ATTN_EXT` | Flash Attention |
| `GET_ROWS` / `SET_ROWS` | Embedding lookup / 写回 |
| `GLU` | SwiGLU/GeGLU/ReGLU/SWIGLU_OAI |
| `RWKV_WKV6/7` | RWKV 专用 |
| `SSM_CONV` / `SSM_SCAN` | Mamba 类状态空间 |
| `GATED_DELTA_NET` / `GATED_LINEAR_ATTN` | 新架构 attention |

训练相关：

| 算子 | 用途 |
|------|------|
| `OPT_STEP_ADAMW` / `OPT_STEP_SGD` | 优化器步进 |
| `CROSS_ENTROPY_LOSS` | 交叉熵损失 |

## 7. 硬件 Backend

| Backend | 平台 | CMake 选项 |
|---------|------|-----------|
| CPU | x86/ARM/RISC-V/PPC/s390/LoongArch/WASM | 默认 ON |
| CUDA | NVIDIA GPU | `GGML_CUDA` |
| HIP / MUSA | AMD / 摩尔线程 | 共用 `ggml-cuda/` |
| Metal | Apple Silicon / macOS | `GGML_METAL`（macOS 默认） |
| Vulkan | 跨平台 GPU | `GGML_VULKAN` |
| SYCL | Intel / 跨厂商 | `GGML_SYCL` |
| WebGPU | 浏览器 / Dawn | `GGML_WEBGPU` |
| RPC | 远程 GPU | `GGML_RPC` |
| VirtGPU | 虚拟 GPU 前后端 | `GGML_VIRTGPU` |
| Hexagon | 高通 NPU | `GGML_HEXAGON` |
| CANN | 华为 Ascend | `GGML_CANN` |
| OpenCL / OpenVINO / BLAS / zDNN | 各厂商 | 条件编译 |

注册顺序（`ggml-backend-reg.cpp`）：CUDA → Metal → SYCL → Vulkan → … → **CPU（最后，兜底）**。

## 8. 典型使用场景

| 场景 | 入口 |
|------|------|
| LLM 推理 | llama.cpp（最大用户） |
| 语音识别 | whisper.cpp |
| 模型量化 | llama-quantize + `quantize_*` |
| 学习张量库 | `examples/simple/` |
| 训练示例 | `examples/` + `ggml-opt` |
| 自定义算子 | `ggml.c` 添加 op + 各 Backend 实现 |

## 9. 与 llama.cpp 关系

```
应用层: llama.cpp
  模型架构、KV cache、采样、OpenAI API
    ↓ ggml_* 算子建图
计算层: ggml
  张量、图、Backend 调度、量化 kernel、gallocr
    ↓
硬件: CPU / CUDA / Metal / Vulkan ...
```

本仓库内嵌副本（对照阅读）：

```
/home/cp/work2/largeModels/03-推理部署/llama.cpp/ggml/
```

## 10. 与本仓库其他模块

| 模块 | 关系 |
|------|------|
| `llama.cpp` / `llama.cppDoc` | 最大用户；KV/batch/loader 文档 |
| `vllm` / `vllmDoc` | 不同栈（HF + PagedAttention）；量化概念对照 |
| `04-量化内核/` | GGML 是本目录核心项目 |

## 推荐阅读顺序

1. 本文 → [02-architecture.md](./02-architecture.md) → [03-tensor-graph.md](./03-tensor-graph.md)
2. 推理内核：[04-backend-scheduler.md](./04-backend-scheduler.md) → [05-memory-alloc.md](./05-memory-alloc.md) → [06-quantization.md](./06-quantization.md) → [07-gguf-format.md](./07-gguf-format.md)
3. 性能：[08-backend-cpu.md](./08-backend-cpu.md) → [09-backend-gpu.md](./09-backend-gpu.md) → [11-build-system.md](./11-build-system.md)
4. 集成：[13-llama-cpp-integration.md](./13-llama-cpp-integration.md)
5. 进阶：[14-ggml-opt-threading.md](./14-ggml-opt-threading.md) → [15-metal-vulkan-deep.md](./15-metal-vulkan-deep.md)

## 上游资源

- 仓库：https://github.com/ggml-org/ggml
- 许可证：MIT
- 官方 docs：`ggml/docs/gguf.md`
