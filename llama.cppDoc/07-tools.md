# 07 - 命令行工具集 (tools/)

## 1. 模块概述

`tools/` 目录包含 llama.cpp 的所有可执行工具，每个子目录对应一个独立工具。工具通过 `common/` 共享基础设施，通过 `libllama` 执行推理。

构建控制: `LLAMA_BUILD_TOOLS=ON` (默认)

## 2. 工具清单

### 2.1 核心工具

| 工具 | 目录 | 可执行文件 | 说明 |
|------|------|-----------|------|
| CLI | `cli/` | `llama-cli` | 交互式/批处理文本生成 |
| Server | `server/` | `llama-server` | OpenAI 兼容 HTTP API 服务 |
| Quantize | `quantize/` | `llama-quantize` | GGUF 模型量化 |
| Bench | `llama-bench/` | `llama-bench` | 性能基准测试 |
| Tokenize | `tokenize/` | `llama-tokenize` | 分词调试工具 |

### 2.2 评估工具

| 工具 | 目录 | 可执行文件 | 说明 |
|------|------|-----------|------|
| Perplexity | `perplexity/` | `llama-perplexity` | 困惑度评估 |
| Batched Bench | `batched-bench/` | `llama-batched-bench` | 批处理性能测试 |
| Completion | `completion/` | `llama-completion` | 批处理补全 |

### 2.3 模型处理

| 工具 | 目录 | 可执行文件 | 说明 |
|------|------|-----------|------|
| GGUF Split | `gguf-split/` | `llama-gguf-split` | 大模型分片/合并 |
| IMatrix | `imatrix/` | `llama-imatrix` | 激活统计 (量化校准) |
| Export LoRA | `export-lora/` | `llama-export-lora` | LoRA 权重导出 |
| Fit Params | `fit-params/` | `llama-fit-params` | 参数拟合 |
| CVector | `cvector-generator/` | `llama-cvector-generator` | Control Vector 生成 |

### 2.4 多模态与特殊

| 工具 | 目录 | 可执行文件 | 说明 |
|------|------|-----------|------|
| MTMD | `mtmd/` | `llama-mtmd-cli` | 多模态 CLI (图像+文本) |
| TTS | `tts/` | `llama-tts` | 文本转语音 |
| UI | `ui/` | (静态资源) | Server Web UI 前端 |
| RPC | `rpc/` | `llama-rpc-server` | 远程 GPU RPC 服务 |

### 2.5 辅助工具

| 工具 | 目录 | 说明 |
|------|------|------|
| Parser | `parser/` | 语法解析测试 |
| Results | `results/` | 结果处理 |

## 3. 核心工具详解

### 3.1 llama-cli

**路径**: `tools/cli/`

主要功能：
- 交互式对话模式
- 单次 prompt 生成
- 从 HuggingFace 直接加载 (`-hf`)
- 支持 LoRA、Control Vector
- 支持投机解码
- 支持多模态输入 (配合 `--mmproj`)

```bash
# 基本用法
llama-cli -m model.gguf -p "Hello, my name is"

# 从 HuggingFace
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# GPU offload
llama-cli -m model.gguf -ngl 99

# 交互模式
llama-cli -m model.gguf -cnv
```

源文件：
- `main.cpp`: 入口
- `cli.cpp`: CLI 逻辑

### 3.2 llama-quantize

**路径**: `tools/quantize/`

将 F16/F32 GGUF 模型量化为低比特格式：

```bash
# 生成 importance matrix (可选, 提高质量)
llama-imatrix -m model-f16.gguf -f calibration.txt -o imatrix.dat

# 量化
llama-quantize --imatrix imatrix.dat model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

支持的量化类型：Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K 到 Q6_K, IQ 系列, TQ 系列等。

源文件：
- `main.cpp`: 入口
- `quantize.cpp`: 量化逻辑

### 3.3 llama-bench

**路径**: `tools/llama-bench/`

性能基准测试，测量：
- Prompt processing (pp) tokens/s
- Text generation (tg) tokens/s
- 不同 batch size、context length、GPU layer 的性能

```bash
llama-bench -m model.gguf -ngl 99 -p 512 -n 128
```

### 3.4 llama-perplexity

**路径**: `tools/perplexity/`

在数据集 (WikiText-2, PTB 等) 上计算困惑度，用于评估模型质量：

```bash
llama-perplexity -m model.gguf -f wiki.test.raw
```

### 3.5 llama-mtmd-cli

**路径**: `tools/mtmd/`

多模态 CLI，支持图像+文本输入：

```bash
llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --image photo.jpg -p "Describe this image"
```

### 3.6 llama-imatrix

**路径**: `tools/imatrix/`

收集模型激活值的统计信息 (importance matrix)，用于提高量化质量：

```bash
llama-imatrix -m model-f16.gguf -f calibration_data.txt -o imatrix.dat -ngl 99
```

## 4. 构建依赖

```
tools/CMakeLists.txt
    |
    +-- batched-bench     -> llama-common
    +-- cli               -> llama-common, server-context (部分)
    +-- server            -> llama-common, mtmd, llama-ui
    +-- quantize          -> llama-common
    +-- llama-bench       -> llama-common
    +-- perplexity        -> llama-common
    +-- mtmd              -> llama-common
    +-- tts               -> llama-common
    +-- ui                -> (前端静态资源)
    +-- rpc               -> llama-common (GGML_RPC)
    +-- cvector-generator -> llama-common (静态 backend)
    +-- export-lora       -> llama-common (静态 backend)
```

Server 构建需要 `LLAMA_BUILD_SERVER=ON` (默认)。

## 5. 安装

```bash
cmake --install build --prefix /usr/local
# 安装: llama-cli, llama-server, llama-quantize, libllama, libggml, headers
```

控制选项: `LLAMA_TOOLS_INSTALL` (默认 ON, iOS 除外)

## 6. 添加新工具

1. 在 `tools/` 下创建子目录
2. 添加 `CMakeLists.txt`:
   ```cmake
   add_executable(my-tool main.cpp)
   target_link_libraries(my-tool PRIVATE llama-common)
   install(TARGETS my-tool RUNTIME)
   ```
3. 在 `tools/CMakeLists.txt` 中添加 `add_subdirectory(my-tool)`
4. 使用 `common_params_parse()` 解析命令行参数
