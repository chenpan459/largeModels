# 01 - 项目总览

## 1. 项目简介

**llama.cpp** 是一个用纯 C/C++ 实现的大语言模型 (LLM) 推理框架。项目最初为在 CPU 上运行 LLaMA 模型而创建，现已发展为支持 100+ 模型架构、多种硬件后端和完整工具链的通用推理平台。

### 核心目标

- 最小依赖：核心库无需 Python/Node 等运行时
- 跨平台：Linux、macOS、Windows、Android、WebAssembly
- 高性能：针对 CPU SIMD、GPU、NPU 深度优化
- 低内存：1-bit 至 8-bit 量化，支持 CPU+GPU 混合卸载

## 2. 项目规模

| 指标 | 数值 |
|------|------|
| 源码目录大小 | ~156 MB |
| C/C++ 源文件 | ~800 个 |
| `src/` 核心代码 | ~69,000 行 |
| 模型架构实现 | 134 个 `.cpp` 文件 |
| GGML 版本 | 0.15.2 |

## 3. 主要产物

| 产物 | 路径 | 说明 |
|------|------|------|
| `libllama` | `src/` + `include/llama.h` | 核心推理库 (C API) |
| `libggml` | `ggml/` | 张量计算引擎 |
| `llama-common` | `common/` | CLI/Server 共享库 |
| `llama-cli` | `tools/cli/` | 命令行推理 |
| `llama-server` | `tools/server/` | OpenAI 兼容 HTTP 服务 |
| `llama-quantize` | `tools/quantize/` | 模型量化工具 |

## 4. 核心特性

### 4.1 硬件支持

- **CPU**: x86 (AVX/AVX2/AVX512/AMX)、ARM (NEON)、RISC-V (RVV)、LoongArch、s390x
- **GPU**: NVIDIA CUDA、AMD HIP、Apple Metal、Vulkan、Intel SYCL
- **其他**: OpenCL、OpenVINO、华为 CANN、摩尔线程 MUSA、高通 Hexagon、WebGPU

### 4.2 模型支持

- **文本生成**: LLaMA 1-4、Qwen 2/3/3.5、DeepSeek、Gemma、Mistral、Phi、GLM 等
- **状态空间模型**: Mamba、Mamba2、RWKV 6/7、Jamba
- **MoE**: Mixtral、DeepSeek-MoE、Qwen3-MoE、DBRX 等
- **多模态**: LLaVA、Qwen2-VL、Gemma3-VL、MiniCPM-V
- **Embedding/Rerank**: BGE、Jina、GritLM
- **特殊**: TTS、Diffusion (LLaDA)、OCR

### 4.3 量化格式

支持 40+ 量化类型，包括：

- 标准: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, BF16
- K-quants: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- Importance: IQ1, IQ2, IQ3, IQ4 系列
- 新格式: MXFP4, NVFP4, TQ1_0, TQ2_0, Q1_0

### 4.4 推理能力

- 连续批处理 (Continuous Batching)
- 投机解码 (Speculative Decoding)
- Flash Attention
- Grammar 约束生成 (GBNF)
- Function Calling / Tool Use
- LoRA / Control Vector 适配
- 多序列并行 (Parallel Sequences)
- KV Cache 量化与卸载

## 5. 目录结构概览

```
llama.cpp/
├── include/           # 公开 C API 头文件
├── src/               # libllama 核心实现
│   └── models/        # 各模型架构的前向图构建
├── ggml/              # 张量计算引擎
├── common/            # CLI/Server 共享工具
├── tools/             # 可执行工具
├── examples/          # 示例程序
├── conversion/        # HF -> GGUF 转换逻辑 (Python)
├── gguf-py/           # Python GGUF 读写库
├── convert_hf_to_gguf.py
├── docs/              # 官方文档
├── tests/             # 单元测试
├── vendor/            # 第三方依赖 (httplib, json)
├── cmake/             # CMake 模块
└── .github/workflows/ # CI/CD
```

## 6. 典型使用场景

| 场景 | 推荐入口 |
|------|----------|
| 本地对话 | `llama-cli` |
| API 服务 | `llama-server` |
| 嵌入应用 | `include/llama.h` C API |
| 模型量化 | `llama-quantize` + `llama-imatrix` |
| HF 模型转换 | `convert_hf_to_gguf.py` |
| 性能测试 | `llama-bench` |
| 学习 API | `examples/simple/` |

## 7. 相关链接

- [推理原理](./18-inference-principles.md) — Prefill/Decode、KV Cache、采样循环
- [官方 README](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
- [构建指南](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [libllama API Changelog](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Server API Changelog](https://github.com/ggml-org/llama.cpp/issues/9291)
