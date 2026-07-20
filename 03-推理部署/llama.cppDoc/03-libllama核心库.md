# 03 - libllama 核心库 (src/)

## 1. 模块概述

`src/` 目录实现 **libllama** 核心推理库，对外通过 `include/llama.h` 暴露 C API。该库负责模型加载、计算图构建、KV cache 管理、采样和分词等全部推理逻辑。

- **构建目标**: `libllama` (CMake target: `llama`)
- **C++ 标准**: C++17
- **代码量**: ~69,000 行 (含 models/)
- **依赖**: `libggml`

### 深度分析文档（本系列扩展）

| 文档 | 专题 |
|------|------|
| [18-推理原理.md](./18-推理原理.md) | 推理原理（Prefill/Decode、KV Cache、采样） |
| [14-模型加载器深度解析.md](./14-模型加载器深度解析.md) | GGUF 加载、create_tensor、mmap |
| [15-Decode流程与Graph复用.md](./15-Decode流程与Graph复用.md) | decode 流程、图复用 can_reuse |
| [16-KV-Cache与Memory系统.md](./16-KV-Cache与Memory系统.md) | KV cache 变体、Memory 工厂 |
| [17-Batch与Micro-batch.md](./17-Batch与Micro-batch.md) | batch 拆分、连续批处理 |

## 2. 源文件清单

### 2.1 核心模块

| 文件 | 行数(约) | 职责 |
|------|----------|------|
| `llama.cpp` | - | C API 入口函数实现 |
| `llama-model.cpp` | 2,713 | 模型类定义、加载、权重管理 |
| `llama-model-loader.cpp` | 1,704 | GGUF 文件解析与张量创建 |
| `llama-model-saver.cpp` | - | 模型保存 (GGUF 写出) |
| `llama-context.cpp` | 4,140 | 推理上下文、decode 主循环 |
| `llama-graph.cpp` | 3,169 | 计算图构建 (通用层操作) |
| `llama-arch.cpp` | - | 架构识别、KV 键映射 |
| `llama-hparams.cpp` | - | 超参数解析与存储 |

### 2.2 内存与缓存

| 文件 | 行数(约) | 职责 |
|------|----------|------|
| `llama-kv-cache.cpp` | 2,632 | 标准 Transformer KV cache |
| `llama-kv-cache-iswa.cpp` | - | Sliding Window Attention KV |
| `llama-kv-cache-dsa.cpp` | - | DeepSeek Sparse Attention KV |
| `llama-memory.cpp` | - | Memory 抽象层工厂 |
| `llama-memory-hybrid.cpp` | - | Attn + SSM 混合记忆 |
| `llama-memory-hybrid-iswa.cpp` | - | Hybrid + ISWA |
| `llama-memory-recurrent.cpp` | - | Mamba/RWKV 循环状态 |
| `llama-kv-cells.h` | - | KV cell 管理与 defrag |

### 2.3 批处理与 I/O

| 文件 | 职责 |
|------|------|
| `llama-batch.cpp` | Batch 分配器 (`llama_batch_allocr`) |
| `llama-cparams.cpp` | 上下文运行时参数 |
| `llama-io.cpp` | 状态序列化/反序列化 |
| `llama-mmap.cpp` | 模型文件内存映射 |

### 2.4 采样与约束

| 文件 | 行数(约) | 职责 |
|------|----------|------|
| `llama-sampler.cpp` | 3,883 | 采样器链 (top-k/p, mirostat, dry, ...) |
| `llama-grammar.cpp` | 1,510 | GBNF 语法约束生成 |
| `llama-vocab.cpp` | 4,333 | 分词器 (BPE/SPM/UGM/WPM/RWKV) |

### 2.5 其他

| 文件 | 职责 |
|------|------|
| `llama-adapter.cpp` | LoRA 适配器、Control Vector |
| `llama-quant.cpp` | 运行时量化相关 |
| `llama-chat.cpp` | Chat 格式辅助 (lib 内部) |
| `llama-impl.cpp` | 内部工具函数 |
| `unicode.cpp` / `unicode-data.cpp` | Unicode 处理 |

## 3. 关键类与结构

### 3.1 llama_model

```cpp
struct llama_model {
    llm_arch arch;                    // 架构类型
    llama_hparams hparams;             // 超参数
    llama_vocab vocab;                 // 分词器
    std::vector<llama_layer> layers;   // 各层权重
    ggml_backend_sched_t sched;        // backend 调度器
    // ...
};
```

职责：
- 从 GGUF 加载权重和 metadata
- 管理 tensor buffer (CPU/GPU)
- 提供架构特定的 graph builder
- 支持 LoRA adapter 加载

### 3.2 llama_context

```cpp
struct llama_context {
    const llama_model & model;
    llama_cparams cparams;
    llama_memory_t memory;
    ggml_backend_sched_t sched;
    // graph, logits, embeddings buffers
};
```

职责：
- 管理 KV cache / 状态记忆
- 执行 decode (build graph -> compute -> update memory)
- 提供 logits 和 embeddings 输出
- 管理采样器链

### 3.3 llama_batch / llama_ubatch

```cpp
struct llama_batch {
    int32_t n_tokens;
    llama_token  * token;
    float        * embd;       // 可选: 直接输入 embedding
    llama_pos    * pos;
    int32_t      * n_seq_id;
    llama_seq_id ** seq_id;
    int8_t       * logits;     // 哪些 token 需要输出 logits
};

struct llama_ubatch {
    // micro-batch: 实际送入 graph 的子批次
    uint32_t n_tokens;
    uint32_t n_seq_tokens;
    uint32_t n_seqs;
    // ...
};
```

`llama_batch_allocr` 将逻辑 batch 拆分为 micro-batch (`ubatch`)，支持连续批处理。

## 4. 推理主流程

### 4.1 初始化

```
llama_backend_init()
llama_model_load_from_file(path, model_params)
  └─ llama_model_loader 解析 GGUF -> create_memory() 选 Memory 类型
llama_init_from_model(model, ctx_params)
  └─ graph_reserve() 预留 worst-case 图 buffer
```

### 4.2 Decode 循环（详见 [15-Decode流程与Graph复用.md](./15-Decode流程与Graph复用.md)）

```
1. llama_batch_get_one() 或手动填充 llama_batch
2. llama_decode(ctx, batch)  ->  llama_context::decode()
   a. balloc->init()           验证 + 自动补全 pos/seq_id/logits
   b. memory->init_batch()     split -> prepare/find_slot -> ubatches[]
   c. loop ubatch:
      i.  mctx->apply()         写入 KV cells
      ii. can_reuse(gparams)?   是 -> set_inputs; 否 -> build_graph
      iii. graph_compute()
   d. output_reorder()          重排输出顺序
3. llama_get_logits(ctx) 或 backend 采样结果
4. llama_sampler_sample(sampler, ctx, idx)
5. 重复 2-4 直到 EOS 或达到 n_predict
```

**decode 返回值**：`0`=成功，`1`=KV 满（非致命），`-2`=内部错误。

### 4.3 状态管理

```
llama_state_save_file()    # 保存 KV cache + rng 状态
llama_state_load_file()    # 恢复状态
llama_kv_cache_clear()     # 清空 KV
llama_kv_cache_seq_rm()    # 删除特定序列
llama_kv_cache_seq_cp()    # 复制序列 (fork)
```

### 4.4 Memory 工厂（详见 [16-KV-Cache与Memory系统.md](./16-KV-Cache与Memory系统.md)）

`llama_model::create_memory()`（`llama-model.cpp:2003`）按架构选择：

| 架构类型 | Memory 实现 |
|----------|-------------|
| BERT/Embedding/LLADA | `nullptr`（encode-only） |
| DeepSeek32 | `llama_kv_cache_dsa`（双 KV） |
| Mamba/RWKV | `llama_memory_recurrent` |
| Jamba/Qwen3.5 等 | `llama_memory_hybrid` / `hybrid_iswa` |
| 标准 Transformer | `llama_kv_cache` / `llama_kv_cache_iswa` |

## 5. 头文件对应关系

| 公开 API | 内部头文件 |
|----------|-----------|
| `include/llama.h` | 全部 C API 声明 |
| `include/llama-cpp.h` | C++ 辅助 wrapper |
| `src/llama-model.h` | 模型类 (内部) |
| `src/llama-context.h` | 上下文类 (内部) |
| `src/llama-graph.h` | 计算图构建 (内部) |
| `src/llama-memory.h` | Memory 接口 (内部) |
| `src/llama-arch.h` | 架构枚举 + KV 键 (内部) |

## 6. 扩展点

| 需求 | 修改位置 |
|------|----------|
| 新增模型架构 | `src/models/` + `llama-arch.h` |
| 新 Memory 类型 | 实现 `llama_memory_i` 接口 |
| 新采样策略 | `llama-sampler.cpp` 添加 sampler |
| 新量化类型 | `llama-quant.cpp` + ggml quants |
| Graph 优化 | `llama-graph.cpp` 通用层 |

## 7. 性能相关

- **图复用**: `llm_graph_result::can_reuse()` 避免重复建图（见 [15-Decode流程与Graph复用.md](./15-Decode流程与Graph复用.md)）
- **Flash Attention**: `-fa on` 启用，减少 KV cache 内存
- **KV 量化**: `--cache-type-k q8_0` 降低 cache 内存
- **Batch 拆分**: `n_ubatch` 控制单次计算 token 数（见 [17-Batch与Micro-batch.md](./17-Batch与Micro-batch.md)）
- **Thread 池**: `n_threads` / `n_threads_batch` 控制并行度
- **mmap**: 默认启用，加速模型加载（见 [14-模型加载器深度解析.md](./14-模型加载器深度解析.md)）
- **异步 GPU 上传**: 非 mmap 模式下 4x64MB pinned buffer 流水线

## 8. 模块间调用关系

```
llama.h (C API)
    |
    v
llama.cpp                    # API -> llama_context/llama_model 转发
    |
    +-- llama_model          # 加载、build_graph、create_memory
    |     +-- llama_model_loader   # GGUF 解析
    |     +-- llama-arch           # 架构识别
    |     +-- models/*.cpp           # 架构特定图
    |
    +-- llama_context        # decode/encode 主循环
    |     +-- llama_batch_allocr   # batch 拆分
    |     +-- llama_memory_i       # KV/recurrent/hybrid
    |     +-- llm_graph_result     # 图 + 复用
    |     +-- ggml_backend_sched     # 多设备调度
    |
    +-- llama_sampler        # 采样器链
    +-- llama_vocab          # 分词器
    +-- llama_grammar        # GBNF 约束
```

## 9. 尚未单独成篇的内部模块

以下模块在 `03-src-core` 清单中列出，后续可继续扩展：

| 文件 | 行数(约) | 专题 |
|------|----------|------|
| `llama-sampler.cpp` | 3,883 | 采样器链、backend 采样、mirostat/dry |
| `llama-vocab.cpp` | 4,333 | BPE/SPM/UGM/WPM/RWKV 分词器 |
| `llama-grammar.cpp` | 1,510 | GBNF 语法约束、偏序集 |
| `llama-adapter.cpp` | 497 | LoRA、Control Vector 注入 |
| `llama-quant.cpp` | 1,407 | 运行时量化、imatrix |
| `llama-io.cpp` | - | 状态序列化协议 |
| `llama-mmap.cpp` | 779 | mmap、direct I/O |
| `unicode*.cpp` | 8,442 | Unicode 规范化（分词依赖） |
