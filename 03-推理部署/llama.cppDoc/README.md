# llama.cpp 项目文档

本目录包含对 `/home/cp/work2/largeModels/03-推理部署/llama.cpp` 项目的结构化分析文档。

## 文档索引

### 概览与架构

| 文档 | 说明 |
|------|------|
| [01-project-overview.md](./01-project-overview.md) | 项目总览、定位与特性 |
| [02-architecture.md](./02-architecture.md) | 整体架构与数据流 |
| [00编译方法.md](./00编译方法.md) | 编译与部署指南 |

### 核心源码深度分析

| 文档 | 说明 |
|------|------|
| [03-src-core.md](./03-src-core.md) | libllama 核心库 (`src/`) 概览 |
| [14-model-loader-deep-dive.md](./14-model-loader-deep-dive.md) | GGUF 加载器、create_tensor、mmap |
| [15-decode-graph-reuse.md](./15-decode-graph-reuse.md) | decode 流程、计算图复用机制 |
| [16-kv-cache-memory.md](./16-kv-cache-memory.md) | KV Cache 变体、Memory 工厂决策树 |
| [17-batch-system.md](./17-batch-system.md) | Batch 拆分、micro-batch、连续批处理 |
| [04-models.md](./04-models.md) | 模型架构实现 (`src/models/`) |
| [05-ggml.md](./05-ggml.md) | GGML 计算引擎与后端 |

### 工具链与 API

| 文档 | 说明 |
|------|------|
| [06-common.md](./06-common.md) | 公共工具库 (`common/`) |
| [07-tools.md](./07-tools.md) | 命令行工具集 (`tools/`) |
| [08-examples.md](./08-examples.md) | 示例程序 (`examples/`) |
| [09-conversion-gguf.md](./09-conversion-gguf.md) | 模型转换与 GGUF 格式 |
| [10-build-system.md](./10-build-system.md) | 构建系统与 CMake 选项 |
| [11-api-reference.md](./11-api-reference.md) | C API 参考概览 |
| [12-server.md](./12-server.md) | llama-server HTTP 服务 |
| [13-tests-ci.md](./13-tests-ci.md) | 测试与 CI |

### 推荐阅读顺序

1. **入门**：01 -> 02 -> 11
2. **理解推理内核**：03 -> 15 -> 16 -> 17 -> 14
3. **添加新模型**：04 -> 09 -> 14
4. **部署服务**：12 -> 06 -> 07

## 项目路径

```
/home/cp/work2/largeModels/03-推理部署/llama.cpp/
```

## 快速参考

```bash
# 构建
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# 推理
./build/bin/llama-cli -m model.gguf

# HTTP 服务
./build/bin/llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## 上游项目

- 仓库: https://github.com/ggml-org/llama.cpp
- 计算库: https://github.com/ggml-org/ggml
- 许可证: MIT
