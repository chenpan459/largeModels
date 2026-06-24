# 08 - 示例程序 (examples/)

## 1. 模块概述

`examples/` 目录包含多个独立的示例程序，演示 libllama C API 的各种用法。这些示例是学习和集成 llama.cpp 的最佳起点。

构建控制: `LLAMA_BUILD_EXAMPLES=ON` (默认)

## 2. 示例清单

| 示例 | 目录 | 说明 | 复杂度 |
|------|------|------|--------|
| simple | `simple/` | 最简推理流程 | 入门 |
| simple-chat | `simple-chat/` | 简单对话循环 | 入门 |
| batched | `batched/` | 批处理推理 | 基础 |
| embedding | `embedding/` | 文本嵌入向量 | 基础 |
| speculative | `speculative/` | 投机解码 | 中级 |
| speculative-simple | `speculative-simple/` | 简化投机解码 | 中级 |
| parallel | `parallel/` | 多序列并行 | 中级 |
| lookup | `lookup/` | N-gram lookup 加速 | 中级 |
| lookahead | `lookahead/` | Lookahead 解码 | 中级 |
| retrieval | `retrieval/` | RAG 检索增强 | 中级 |
| training | `training/` | 微调/训练示例 | 高级 |
| diffusion | `diffusion/` | Diffusion LM 推理 | 高级 |
| passkey | `passkey/` | Passkey 检索测试 | 测试 |
| eval-callback | `eval-callback/` | 评估回调 | 高级 |
| debug | `debug/` | 调试工具 | 工具 |
| idle | `idle/` | 空闲回调 | 基础 |
| gguf | `gguf/` | GGUF 文件操作 | 基础 |
| gguf-hash | `gguf-hash/` | GGUF 哈希校验 | 工具 |
| gen-docs | `gen-docs/` | 文档生成 | 工具 |
| convert-llama2c-to-ggml | `convert-llama2c-to-ggml/` | 旧格式转换 | 工具 |

### 平台特定示例

| 示例 | 目录 | 说明 |
|------|------|------|
| llama.android | `llama.android/` | Android 集成 |
| llama.swiftui | `llama.swiftui/` | SwiftUI macOS/iOS |
| batched.swift | `batched.swift/` | Swift 批处理 |
| sycl | `sycl/` | Intel SYCL 示例 |
| llama.vim | `llama.vim/` | Vim/Neovim 插件 |
| model-conversion | `model-conversion/` | 模型转换示例 |
| llama-eval | `llama-eval/` | 评估框架 |

## 3. 重点示例详解

### 3.1 simple - 最简推理

**路径**: `examples/simple/simple.cpp`

演示完整的推理流程：

```cpp
// 1. 初始化 backend
llama_backend_init();

// 2. 加载模型
llama_model * model = llama_model_load_from_file(path, model_params);

// 3. 创建 context
llama_context * ctx = llama_init_from_model(model, ctx_params);

// 4. Tokenize
llama_tokenize(vocab, prompt, ..., tokens, ...);

// 5. Decode loop
while (n_cur <= n_len) {
    llama_decode(ctx, batch);
    llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
    // ...
}

// 6. 清理
llama_free(ctx);
llama_model_free(model);
llama_backend_free();
```

关键参数：
- `-m`: 模型路径
- `-n`: 生成 token 数
- `-ngl`: GPU 层数

### 3.2 simple-chat - 对话循环

**路径**: `examples/simple-chat/simple-chat.cpp`

在 simple 基础上添加：
- 多轮对话 (维护 chat history)
- Chat 模板应用
- 交互式输入

### 3.3 embedding - 文本嵌入

**路径**: `examples/embedding/embedding.cpp`

演示如何获取文本的 embedding 向量：

```cpp
// pooling_type 设为 MEAN 或 CLS
ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

llama_decode(ctx, batch);
float * embd = llama_get_embeddings_seq(ctx, seq_id);
```

适用于 RAG、语义搜索等场景。

### 3.4 batched - 批处理

**路径**: `examples/batched/batched.cpp`

演示同时处理多个序列：

```cpp
llama_batch batch = llama_batch_init(n_tokens, 0, n_seqs);
// 为每个序列设置不同的 seq_id
batch.seq_id[i][0] = seq_id;
llama_decode(ctx, batch);
```

### 3.5 speculative - 投机解码

**路径**: `examples/speculative/speculative.cpp`

使用小模型加速大模型推理：

```cpp
// 加载 draft model 和 target model
llama_model * model_dft = ...;
llama_model * model_tgt = ...;

// draft model 生成候选 tokens
// target model 批量验证
// 接受匹配的 tokens
```

### 3.6 parallel - 多序列并行

**路径**: `examples/parallel/parallel.cpp`

在单个 context 中并行处理多个独立对话：

```cpp
// n_seq_max > 1
ctx_params.n_seq_max = 4;

// 每个序列独立维护 KV cache
llama_kv_cache_seq_cp(ctx, 0, 1, -1, -1);  // fork 序列
```

### 3.7 training - 微调

**路径**: `examples/training/training.cpp`

演示使用 ggml 优化器进行 LoRA 微调：

```cpp
// 使用 ggml_opt 进行梯度下降
ggml_opt_dataset_t dataset = ...;
ggml_opt(ctx, ...);
```

## 4. 构建与运行

```bash
# 构建所有示例
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --config Release

# 运行
./build/bin/llama-simple -m model.gguf -p "Hello"
./build/bin/llama-simple-chat -m model.gguf
./build/bin/llama-embedding -m embed-model.gguf -p "text to embed"
```

## 5. 集成指南

### 5.1 CMake 集成

```cmake
find_package(llama REQUIRED)
add_executable(my-app main.cpp)
target_link_libraries(my-app PRIVATE llama)
```

或使用 `examples/simple-cmake-pkg/` 作为参考。

### 5.2 最小集成步骤

1. 参考 `examples/simple/simple.cpp`
2. 链接 `libllama` 和 `libggml`
3. 包含 `llama.h`
4. 调用 backend_init -> model_load -> init_context -> decode loop

### 5.3 关键注意事项

- 线程安全: 每个 `llama_context` 不应跨线程共享
- 内存管理: 模型可共享，context 需独立
- Batch 大小: `n_batch` 影响内存和性能
- GPU offload: 通过 `model_params.n_gpu_layers` 控制

## 6. 辅助脚本

| 文件 | 说明 |
|------|------|
| `json_schema_to_grammar.py` | JSON Schema 转 GBNF |
| `pydantic_models_to_grammar.py` | Pydantic 模型转 Grammar |
| `regex_to_grammar.py` | 正则转 Grammar |
| `reason-act.sh` | Reasoning + Action 示例 |
| `server_embd.py` | Server embedding 示例 |
| `convert_legacy_llama.py` | 旧版模型转换 |
