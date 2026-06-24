# 06 - 公共工具库 (common/)

## 1. 模块概述

`common/` 目录实现 **llama-common** 共享库，为 CLI、Server 和其他工具提供统一的基础设施。几乎所有用户-facing 工具都依赖此库。

- **构建目标**: `llama-common` (动态/静态库)
- **依赖**: `libllama`, `cpp-httplib`, `Threads`
- **C++ 标准**: C++17

## 2. 源文件清单

### 2.1 命令行与配置

| 文件 | 行数(约) | 职责 |
|------|----------|------|
| `arg.cpp` / `arg.h` | 4,156 | 命令行参数解析 (所有工具共享) |
| `common.cpp` / `common.h` | 2,155 | 通用初始化、模型加载封装 |
| `preset.cpp` / `preset.h` | - | 预设配置管理 |
| `console.cpp` / `console.h` | - | 终端交互 (颜色、readline) |
| `log.cpp` / `log.h` | - | 日志系统 |

### 2.2 Chat 与模板

| 文件 | 行数(约) | 职责 |
|------|----------|------|
| `chat.cpp` / `chat.h` | 2,762 | Chat 模板应用 (Jinja 风格) |
| `chat-diff-analyzer.cpp` | 1,577 | Chat diff 分析 |
| `chat-auto-parser-generator.cpp` | - | 自动 parser 生成 |
| `chat-auto-parser-helpers.cpp` | - | Auto parser 辅助 |
| `chat-peg-parser.cpp` / `.h` | - | PEG parser for chat |
| `jinja/` | - | 内嵌 Jinja2 模板引擎 |

Jinja 子模块：

| 文件 | 职责 |
|------|------|
| `jinja/lexer.cpp` | 词法分析 |
| `jinja/parser.cpp` | 语法分析 |
| `jinja/runtime.cpp` | 模板渲染 |
| `jinja/value.cpp` | 值类型 |
| `jinja/string.cpp` | 字符串处理 |
| `jinja/caps.cpp` | 能力检测 |

### 2.3 采样与推理

| 文件 | 行数(约) | 职责 |
|------|----------|------|
| `sampling.cpp` / `sampling.h` | - | 采样参数封装 |
| `speculative.cpp` / `speculative.h` | 2,304 | 投机解码实现 |
| `reasoning-budget.cpp` / `.h` | - | 推理预算控制 (thinking models) |

### 2.4 Grammar 与结构化输出

| 文件 | 职责 |
|------|------|
| `json-schema-to-grammar.cpp` | JSON Schema -> GBNF grammar |
| `peg-parser.cpp` / `.h` | PEG 语法解析器 (2,256 行) |
| `regex-partial.cpp` / `.h` | 部分正则匹配 |
| `json-partial.cpp` / `.h` | 部分 JSON 解析 |
| `llguidance.cpp` | LLGuidance 集成 (可选) |

### 2.5 模型下载与管理

| 文件 | 职责 |
|------|------|
| `download.cpp` / `download.h` | 模型下载 (URL/HF) |
| `hf-cache.cpp` / `hf-cache.h` | HuggingFace 缓存管理 |
| `imatrix-loader.cpp` / `.h` | Importance matrix 加载 |
| `http.h` | HTTP 客户端封装 |

### 2.6 N-gram 缓存

| 文件 | 职责 |
|------|------|
| `ngram-cache.cpp` / `.h` | N-gram 缓存 (lookup 加速) |
| `ngram-map.cpp` / `.h` | N-gram 映射 |
| `ngram-mod.cpp` / `.h` | N-gram 修改 |

### 2.7 其他

| 文件 | 职责 |
|------|------|
| `fit.cpp` / `fit.h` | 参数拟合 |
| `debug.cpp` / `debug.h` | 调试工具 |
| `unicode.cpp` / `.h` | Unicode 处理 |
| `base64.hpp` | Base64 编解码 |
| `build-info.cpp.in` | 构建信息模板 |

## 3. 核心功能详解

### 3.1 命令行参数系统 (arg.cpp)

所有工具 (cli, server, quantize, bench, ...) 共享统一的参数定义：

```cpp
// 典型用法
common_params params;
if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
    return 1;
}
common_init_result llama_init = common_init_from_params(params);
```

支持的参数类别：
- 模型: `-m`, `-hf`, `--model-url`
- 推理: `-n`, `-c`, `-b`, `-t`, `--temp`, `--top-k`, `--top-p`
- GPU: `-ngl`, `-dev`, `--split-mode`
- 采样: `--mirostat`, `--dry-*`, `--xtc-*`
- LoRA: `--lora`, `--control-vector`
- 多模态: `--mmproj`, `--image`

### 3.2 Chat 模板 (chat.cpp)

将 OpenAI 格式的 messages 转换为模型特定的 prompt：

```
Input (OpenAI format):
  [{"role": "system", "content": "..."},
   {"role": "user", "content": "Hello"}]

    |
    v  apply_template()

Output (model-specific):
  "<|im_start|>system\n...\n<|im_start|>user\nHello\n<|im_start|>assistant\n"
```

支持：
- Jinja2 模板 (从 GGUF metadata 或外部文件加载)
- Tool/Function calling 格式
- Reasoning/Thinking 格式 (DeepSeek-R1, QwQ)
- 多模态 content (text + image)

### 3.3 投机解码 (speculative.cpp)

```
Draft Model (小模型)          Target Model (大模型)
    |                              |
    v                              v
生成 N 个 draft tokens    一次验证 N 个 tokens
    |                              |
    +-------- 接受/拒绝 ----------+
                  |
                  v
           输出接受的 tokens
```

### 3.4 HuggingFace 集成 (hf-cache.cpp)

```bash
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

- 自动从 HuggingFace Hub 下载 GGUF 文件
- 使用标准 HF 缓存目录 (`~/.cache/huggingface/`)
- 支持 `--hf-repo`, `--hf-file` 精确指定

## 4. 依赖关系

```
llama-common
    |
    +-- llama (libllama)
    +-- llama-common-base (build info)
    +-- cpp-httplib (HTTP client)
    +-- Threads
    +-- llguidance (可选, LLAMA_LLGUIDANCE=ON)
```

## 5. 使用此库的工具

- `llama-cli` (tools/cli/)
- `llama-server` (tools/server/)
- `llama-quantize` (tools/quantize/)
- `llama-bench` (tools/llama-bench/)
- `llama-perplexity` (tools/perplexity/)
- `llama-completion` (tools/completion/)
- 所有 examples/

## 6. 扩展指南

| 需求 | 修改位置 |
|------|----------|
| 新增 CLI 参数 | `arg.cpp` 添加参数定义 |
| 新 Chat 模板格式 | `chat.cpp` 或外部 Jinja 模板 |
| 新采样策略默认值 | `sampling.cpp` |
| JSON Schema 支持 | `json-schema-to-grammar.cpp` |
| 新下载源 | `download.cpp` |
