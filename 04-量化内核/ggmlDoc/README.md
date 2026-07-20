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
| [01-GGML项目总览.md](./01-GGML项目总览.md) | 目录结构、库拆分、42 类型、88 算子 |
| [02-整体架构.md](./02-整体架构.md) | 分层架构、数据流、五层 Backend 接口 |
| [03-张量模型与计算图.md](./03-张量模型与计算图.md) | ggml_tensor、cgraph、use_counts、建图 API |

### 核心模块

| 文档 | 说明 |
|------|------|
| [04-Backend抽象与调度器.md](./04-Backend抽象与调度器.md) | sched 三 Pass、pipeline、async/event |
| [05-内存分配器.md](./05-内存分配器.md) | gallocr、hash_node、in-place 生命周期 |
| [06-量化系统.md](./06-量化系统.md) | block 格式、vec_dot 矩阵、repack、imatrix |
| [07-GGUF文件格式.md](./07-GGUF文件格式.md) | v3 布局、多入口加载、写入 API |

### 硬件 Backend

| 文档 | 说明 |
|------|------|
| [08-CPU后端.md](./08-CPU后端.md) | ops.cpp、arch SIMD、AMX、extra buffer |
| [09-GPU后端.md](./09-GPU后端.md) | CUDA 65 kernel 分类、Metal/Vulkan |
| [10-其他后端.md](./10-其他后端.md) | SYCL/HIP/RPC/WebGPU/VirtGPU/CANN |
| [15-Metal与Vulkan深度解析.md](./15-Metal与Vulkan深度解析.md) | **Metal 分层、Vulkan shader-gen** |

### 构建、API 与集成

| 文档 | 说明 |
|------|------|
| [11-构建系统.md](./11-构建系统.md) | CMake 选项、GGML_SCHED_MAX_COPIES |
| [12-API参考与速查.md](./12-API参考与速查.md) | API 速查、行号锚点 |
| [13-与llama.cpp集成.md](./13-与llama.cpp集成.md) | llama 调用链、量化工具链 |
| [14-ggml-opt与线程系统.md](./14-ggml-opt与线程系统.md) | **训练 opt、线程池** |

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
4. **llama 对照**：13 → `03-推理部署/llama.cppDoc/05-GGML计算引擎.md`
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
