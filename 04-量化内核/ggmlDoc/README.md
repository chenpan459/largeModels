# GGML 项目文档

本目录包含对 `/home/cp/work2/largeModels/04-量化内核/ggml` 项目的结构化源码分析文档（深度扩展版）。

## 项目定位

**GGML** 是 llama.cpp 的底层 C/C++ 张量计算引擎：惰性建图、42+ 量化类型、多 Backend 调度、GGUF 格式、图级 in-place 内存复用。

- 上游：https://github.com/ggml-org/ggml
- 版本：**0.15.3**
- 许可证：MIT

## 文档索引

### 概览与架构

| 文档 | 说明 |
|------|------|
| [01-project-overview.md](./01-project-overview.md) | 目录结构、库拆分、42 类型、88 算子 |
| [02-architecture.md](./02-architecture.md) | 分层架构、数据流、五层 Backend 接口 |
| [03-tensor-graph.md](./03-tensor-graph.md) | ggml_tensor、cgraph、use_counts、建图 API |

### 核心模块

| 文档 | 说明 |
|------|------|
| [04-backend-scheduler.md](./04-backend-scheduler.md) | sched 三 Pass、pipeline、async/event |
| [05-memory-alloc.md](./05-memory-alloc.md) | gallocr、hash_node、in-place 生命周期 |
| [06-quantization.md](./06-quantization.md) | block 格式、vec_dot 矩阵、repack、imatrix |
| [07-gguf-format.md](./07-gguf-format.md) | v3 布局、多入口加载、写入 API |

### 硬件 Backend

| 文档 | 说明 |
|------|------|
| [08-backend-cpu.md](./08-backend-cpu.md) | ops.cpp、arch SIMD、AMX、extra buffer |
| [09-backend-gpu.md](./09-backend-gpu.md) | CUDA 65 kernel 分类、Metal/Vulkan |
| [10-other-backends.md](./10-other-backends.md) | SYCL/HIP/RPC/WebGPU/VirtGPU/CANN |
| [15-metal-vulkan-deep.md](./15-metal-vulkan-deep.md) | **Metal 分层、Vulkan shader-gen** |

### 构建、API 与集成

| 文档 | 说明 |
|------|------|
| [11-build-system.md](./11-build-system.md) | CMake 选项、GGML_SCHED_MAX_COPIES |
| [12-api-reference.md](./12-api-reference.md) | API 速查、行号锚点 |
| [13-llama-cpp-integration.md](./13-llama-cpp-integration.md) | llama 调用链、量化工具链 |
| [14-ggml-opt-threading.md](./14-ggml-opt-threading.md) | **训练 opt、线程池** |

## 项目路径

```
/home/cp/work2/largeModels/04-量化内核/ggml/
├── include/          # ggml.h, ggml-backend.h, gguf.h
├── src/              # ggml.c, ggml-backend.cpp, ggml-alloc.c, ggml-quants.c, gguf.cpp
│   ├── ggml-cpu/     # CPU Backend + arch/*
│   ├── ggml-cuda/    # 65 .cu
│   ├── ggml-metal/   # 多文件 Metal
│   └── ggml-vulkan/  # shader-gen + 132+ comp
└── tests/

llama.cpp 内嵌副本：
/home/cp/work2/largeModels/03-推理部署/llama.cpp/ggml/
```

## 推荐阅读顺序

1. **入门**：01 → 02 → 03 → 12
2. **推理内核**：04 → 05 → 06 → 07
3. **性能**：08 → 09 → 15 → 11
4. **llama 对照**：13 → `03-推理部署/llama.cppDoc/05-ggml.md`
5. **训练/线程**：14

## 快速开始

```bash
cd /home/cp/work2/largeModels/04-量化内核/ggml
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
```

## 与本仓库其他模块

| 模块 | 关系 |
|------|------|
| `llama.cpp` / `llama.cppDoc` | 最大用户；应用层文档 |
| `vllm` / `vllmDoc` | 不同推理栈；量化概念对照 |
| `04-量化内核/` | GGML 为本目录核心 |

## 文档版本说明

基于本地 GGML 0.15.3 源码分析，覆盖 gallocr 实际结构、vec_dot 体系、Metal 多文件架构、Vulkan shader-gen、callback GGUF 加载、ggml-opt/threading 等。upstream 升级请以源码为准。
