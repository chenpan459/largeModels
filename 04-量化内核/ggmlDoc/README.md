# GGML 项目文档

本目录包含对 `/home/cp/work2/largeModels/04-量化内核/ggml` 项目的结构化源码分析文档。

## 文档索引

### 概览与架构

| 文档 | 说明 |
|------|------|
| [01-project-overview.md](./01-project-overview.md) | 项目总览、定位与版本 |
| [02-architecture.md](./02-architecture.md) | 分层架构与数据流 |

### 核心模块

| 文档 | 说明 |
|------|------|
| [03-tensor-graph.md](./03-tensor-graph.md) | 张量模型与计算图 (`ggml.c`) |
| [04-backend-scheduler.md](./04-backend-scheduler.md) | Backend 抽象与调度器 |
| [05-memory-alloc.md](./05-memory-alloc.md) | 内存分配器 (`ggml-alloc.c`) |
| [06-quantization.md](./06-quantization.md) | 量化格式与内核 |
| [07-gguf-format.md](./07-gguf-format.md) | GGUF 文件格式 |

### 硬件后端

| 文档 | 说明 |
|------|------|
| [08-backend-cpu.md](./08-backend-cpu.md) | CPU 后端 (SIMD/AMX/Repack) |
| [09-backend-gpu.md](./09-backend-gpu.md) | CUDA / Metal / Vulkan 等 GPU 后端 |
| [10-other-backends.md](./10-other-backends.md) | SYCL/HIP/RPC/WebGPU 等 |

### 构建与集成

| 文档 | 说明 |
|------|------|
| [11-build-system.md](./11-build-system.md) | CMake 构建选项 |
| [12-api-reference.md](./12-api-reference.md) | 公共 C API 概览 |
| [13-llama-cpp-integration.md](./13-llama-cpp-integration.md) | 与 llama.cpp 的集成关系 |

## 项目路径

```
/home/cp/work2/largeModels/04-量化内核/ggml/
```

llama.cpp 内嵌副本（对照阅读）：

```
/home/cp/work2/largeModels/03-推理部署/llama.cpp/ggml/
```

## 推荐阅读顺序

1. **入门**：01 -> 02 -> 03
2. **推理内核**：04 -> 05 -> 06 -> 07
3. **性能优化**：08 -> 09 -> 11
4. **与 llama.cpp 对照**：13 -> `03-推理部署/llama.cppDoc/05-ggml.md`

## 快速参考

```bash
# 构建（CPU only）
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# 版本
# GGML 0.15.3 (CMakeLists.txt)
```

## 上游项目

- 仓库: https://github.com/ggml-org/ggml
- 许可证: MIT
