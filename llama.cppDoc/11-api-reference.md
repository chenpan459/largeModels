# 11 - C API 参考概览

## 1. 概述

llama.cpp 通过 `include/llama.h` 暴露 **C 语言 API**（约 1,590 行），可在 C/C++/Python (ctypes)/Rust (bindgen) 等语言中调用。C++ 辅助头文件 `include/llama-cpp.h` 提供 RAII wrapper。

## 2. API 模块分类

### 2.1 初始化与销毁

```c
// Backend 全局初始化 (程序开始时调用一次)
LLAMA_API void llama_backend_init(void);
LLAMA_API void llama_backend_free(void);

// NUMA 优化
LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);

// 默认参数
LLAMA_API struct llama_model_params   llama_model_default_params(void);
LLAMA_API struct llama_context_params llama_context_default_params(void);
```

### 2.2 模型加载

```c
// 从文件加载
LLAMA_API struct llama_model * llama_model_load_from_file(
    const char * path,
    struct llama_model_params params);

// 从 HuggingFace 加载
LLAMA_API struct llama_model * llama_model_init_from_user(
    const char * user,   // HF repo id
    const char * token,  // HF token (可选)
    struct llama_model_params params);

// 从分片加载
LLAMA_API struct llama_model * llama_model_load_from_splits(
    const char ** paths, size_t n_paths,
    struct llama_model_params params);

// 释放
LLAMA_API void llama_model_free(struct llama_model * model);
```

**llama_model_params** 关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `n_gpu_layers` | int32_t | GPU offload 层数 (-1=全部) |
| `split_mode` | enum | 多 GPU 分割模式 |
| `main_gpu` | int32_t | 主 GPU 索引 |
| `tensor_split` | float* | 各 GPU 负载比例 |
| `use_mmap` | bool | 内存映射 |
| `use_mlock` | bool | 锁定内存 |
| `vocab_only` | bool | 仅加载词表 |
| `devices` | ggml_backend_dev_t* | 指定设备 |

### 2.3 上下文管理

```c
// 创建推理上下文
LLAMA_API struct llama_context * llama_init_from_model(
    struct llama_model * model,
    struct llama_context_params params);

// 释放
LLAMA_API void llama_free(struct llama_context * ctx);

// 查询
LLAMA_API uint32_t llama_n_ctx(const struct llama_context * ctx);
LLAMA_API uint32_t llama_n_batch(const struct llama_context * ctx);
```

**llama_context_params** 关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `n_ctx` | uint32_t | 上下文长度 |
| `n_batch` | uint32_t | 最大 batch size |
| `n_ubatch` | uint32_t | micro-batch size |
| `n_seq_max` | uint32_t | 最大并行序列数 |
| `n_threads` | int32_t | 生成线程数 |
| `n_threads_batch` | int32_t | 批处理线程数 |
| `rope_scaling_type` | enum | RoPE 缩放 (YARN/LongRoPE) |
| `pooling_type` | enum | 池化类型 (embedding) |
| `flash_attn_type` | enum | Flash Attention |
| `type_k` / `type_v` | enum | KV cache 数据类型 |
| `offload_kqv` | bool | KV cache GPU offload |

### 2.4 分词

```c
// Tokenize
LLAMA_API int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text, int32_t text_len,
    llama_token * tokens, int32_t n_tokens_max,
    bool add_special, bool parse_special);

// Detokenize
LLAMA_API int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token, char * buf, int32_t length,
    int32_t lstrip, bool special);

// 特殊 token
LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab);
LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab);
LLAMA_API llama_token llama_vocab_nl(const struct llama_vocab * vocab);
```

### 2.5 推理 (Decode)

```c
// 核心 decode 函数
LLAMA_API int32_t llama_decode(
    struct llama_context * ctx,
    struct llama_batch batch);

// 获取输出
LLAMA_API float * llama_get_logits(struct llama_context * ctx);
LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);
```

**llama_batch** 结构：

```c
struct llama_batch {
    int32_t n_tokens;
    llama_token  * token;     // token IDs
    float        * embd;      // 或直接 embedding 输入
    llama_pos    * pos;       // 位置
    int32_t      * n_seq_id;  // 每个 token 的 seq_id 数
    llama_seq_id ** seq_id;   // 序列 ID
    int8_t       * logits;    // 是否输出 logits (0/1)
};
```

辅助函数：

```c
// 创建单 token batch
LLAMA_API struct llama_batch llama_batch_get_one(
    llama_token * tokens, int32_t n_tokens);

// 初始化/释放 batch
LLAMA_API struct llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
LLAMA_API void llama_batch_free(struct llama_batch batch);
```

### 2.6 采样

```c
// 采样器链接口
struct llama_sampler;

// 创建采样器链
LLAMA_API struct llama_sampler * llama_sampler_chain_init(
    struct llama_sampler_chain_params params);

// 添加采样策略
LLAMA_API void llama_sampler_chain_add(
    struct llama_sampler * chain,
    struct llama_sampler * smpl);

// 采样
LLAMA_API llama_token llama_sampler_sample(
    struct llama_sampler * smpl,
    struct llama_context * ctx,
    int32_t idx);

// 接受 token (更新 grammar 等)
LLAMA_API void llama_sampler_accept(
    struct llama_sampler * smpl,
    llama_token token);

// 内置采样器
LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void);
LLAMA_API struct llama_sampler * llama_sampler_init_top_k(int32_t k);
LLAMA_API struct llama_sampler * llama_sampler_init_top_p(float p, size_t min_keep);
LLAMA_API struct llama_sampler * llama_sampler_init_min_p(float p, size_t min_keep);
LLAMA_API struct llama_sampler * llama_sampler_init_temp(float t);
LLAMA_API struct llama_sampler * llama_sampler_init_dist(uint32_t seed);
LLAMA_API struct llama_sampler * llama_sampler_init_mirostat(...);
LLAMA_API struct llama_sampler * llama_sampler_init_penalties(...);
LLAMA_API struct llama_sampler * llama_sampler_init_grammar(...);
LLAMA_API struct llama_sampler * llama_sampler_init_dry(...);
```

### 2.7 KV Cache 管理

```c
// 序列操作
LLAMA_API void llama_kv_cache_clear(struct llama_context * ctx);
LLAMA_API bool llama_kv_cache_seq_rm(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
LLAMA_API void llama_kv_cache_seq_cp(struct llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
LLAMA_API void llama_kv_cache_seq_keep(struct llama_context * ctx, llama_seq_id seq_id);
LLAMA_API void llama_kv_cache_seq_add(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta);
LLAMA_API void llama_kv_cache_seq_div(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d);
```

### 2.8 状态保存/加载

```c
LLAMA_API size_t llama_state_get_size(struct llama_context * ctx);
LLAMA_API size_t llama_state_get_data(struct llama_context * ctx, uint8_t * dst, size_t size);
LLAMA_API size_t llama_state_set_data(struct llama_context * ctx, const uint8_t * src, size_t size);
LLAMA_API bool llama_state_load_file(struct llama_context * ctx, const char * path, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
LLAMA_API bool llama_state_save_file(struct llama_context * ctx, const char * path, const llama_token * tokens, size_t n_token_count);
```

### 2.9 LoRA 适配器

```c
LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init(
    struct llama_model * model, const char * path);
LLAMA_API void llama_adapter_lora_free(struct llama_adapter_lora * adapter);
LLAMA_API int32_t llama_set_adapter_lora(
    struct llama_context * ctx, struct llama_adapter_lora * adapter, float scale);
```

### 2.10 模型信息

```c
LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);
LLAMA_API uint64_t llama_model_size(const struct llama_model * model);
LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);
LLAMA_API int32_t llama_model_n_embd(const struct llama_model * model);
LLAMA_API int32_t llama_model_n_layer(const struct llama_model * model);
LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);
```

## 3. 完整推理示例

```c
#include "llama.h"
#include <stdio.h>

int main() {
    llama_backend_init();

    // 加载模型
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    struct llama_model * model = llama_model_load_from_file("model.gguf", model_params);

    // 创建 context
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    struct llama_context * ctx = llama_init_from_model(model, ctx_params);

    // 采样器
    struct llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Tokenize
    const char * prompt = "Hello, my name is";
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token tokens[256];
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                   tokens, 256, true, true);

    // Decode prompt
    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    llama_decode(ctx, batch);

    // Generate
    for (int i = 0; i < 32; i++) {
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, new_token);

        if (new_token == llama_vocab_eos(vocab)) break;

        char piece[32];
        llama_token_to_piece(vocab, new_token, piece, 32, 0, true);
        printf("%s", piece);
        fflush(stdout);

        batch = llama_batch_get_one(&new_token, 1);
        llama_decode(ctx, batch);
    }

    // 清理
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
```

## 4. 关键枚举

### llama_ftype (量化类型)

F32=0, F16=1, Q4_0=2, Q4_K_M=15, Q6_K=18, IQ4_NL=25, MXFP4=38, ...

### llama_rope_scaling_type

NONE=0, LINEAR=1, YARN=2, LONGROPE=3

### llama_pooling_type

NONE=0, MEAN=1, CLS=2, LAST=3, RANK=4

### llama_split_mode

NONE=0 (单 GPU), LAYER=1, ROW=2, TENSOR=3

## 5. 线程安全

- `llama_model` 可在多线程间共享 (只读)
- `llama_context` **不应**跨线程共享
- 每个线程应创建独立的 context
- `llama_backend_init()` / `free()` 非线程安全

## 6. API 变更追踪

- libllama API: https://github.com/ggml-org/llama.cpp/issues/9289
- Server REST API: https://github.com/ggml-org/llama.cpp/issues/9291

## 7. 头文件

| 文件 | 说明 |
|------|------|
| `include/llama.h` | 主 C API |
| `include/llama-cpp.h` | C++ RAII wrapper |
| `ggml/include/ggml.h` | GGML 张量 API |
| `ggml/include/ggml-backend.h` | Backend API |
| `ggml/include/gguf.h` | GGUF 格式 API |
