# llama.cpp 项目文档

本目录包含对 `/home/cp/work2/largeModels/03-推理部署/llama.cpp` 项目的结构化分析文档。

## 文档索引

### 概览与架构

| 文档 | 说明 |
|------|------|
| [01-llama.cpp项目总览.md](./01-llama.cpp项目总览.md) | 项目总览、定位与特性 |
| [18-推理原理.md](./18-推理原理.md) | **推理原理**（Prefill/Decode、KV Cache、采样循环） |
| [02-整体架构.md](./02-整体架构.md) | 整体架构与数据流 |
| [00-编译与测试指南.md](./00-编译与测试指南.md) | 编译与部署指南 |

### 核心源码深度分析

| 文档 | 说明 |
|------|------|
| [03-libllama核心库.md](./03-libllama核心库.md) | libllama 核心库 (`src/`) 概览 |
| [14-模型加载器深度解析.md](./14-模型加载器深度解析.md) | GGUF 加载器、create_tensor、mmap |
| [15-Decode流程与Graph复用.md](./15-Decode流程与Graph复用.md) | decode 流程、计算图复用机制 |
| [16-KV-Cache与Memory系统.md](./16-KV-Cache与Memory系统.md) | KV Cache 变体、Memory 工厂决策树 |
| [17-Batch与Micro-batch.md](./17-Batch与Micro-batch.md) | Batch 拆分、micro-batch、连续批处理 |
| [04-模型架构层.md](./04-模型架构层.md) | 模型架构实现 (`src/models/`) |
| [05-GGML计算引擎.md](./05-GGML计算引擎.md) | GGML 计算引擎与后端 |

### 工具链与 API

| 文档 | 说明 |
|------|------|
| [06-公共工具库.md](./06-公共工具库.md) | 公共工具库 (`common/`) |
| [07-命令行工具集.md](./07-命令行工具集.md) | 命令行工具集 (`tools/`) |
| [08-示例程序.md](./08-示例程序.md) | 示例程序 (`examples/`) |
| [09-模型转换与GGUF格式.md](./09-模型转换与GGUF格式.md) | 模型转换与 GGUF 格式 |
| [10-构建系统.md](./10-构建系统.md) | 构建系统与 CMake 选项 |
| [11-C-API参考.md](./11-C-API参考.md) | C API 参考概览 |
| [12-llama-server服务.md](./12-llama-server服务.md) | llama-server HTTP 服务 |
| [13-测试与CI.md](./13-测试与CI.md) | 测试与 CI |

### 推荐阅读顺序

1. **入门**：01 -> **18** -> 02 -> 11
2. **理解推理内核**：18 -> 03 -> 15 -> 16 -> 17 -> 14
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
