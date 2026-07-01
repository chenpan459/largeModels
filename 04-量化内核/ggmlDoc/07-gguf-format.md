# 07 - GGUF 文件格式

## 1. 模块概述

| 文件 | 行数 | 职责 |
|------|------|------|
| `include/gguf.h` | ~210 | 公共 GGUF API |
| `src/gguf.cpp` | ~1,688 | 读写实现 |

GGUF (GGML Unified Format) 是自描述模型文件格式，llama.cpp 的标准模型容器。

---

## 2. 文件布局

```
+-------------------+
| Header            |
|  magic: "GGUF"    |  4 bytes
|  version: uint32  |  当前 v3
|  n_tensors        |  uint64
|  n_kv             |  uint64
+-------------------+
| Metadata KV pairs |
|  for each KV:     |
|    key: string    |
|    type: enum     |
|    value          |
+-------------------+
| Tensor Info       |
|  for each tensor: |
|    name: string   |
|    n_dims: uint32 |
|    dims: uint64[] |
|    type: enum     |
|    offset: uint64 |
+-------------------+
| Tensor Data       |
|  raw bytes        |
|  (alignment pad)  |
+-------------------+
```

---

## 3. 核心结构（`gguf.cpp`）

### 3.1 `gguf_kv`

| 字段 | 说明 |
|------|------|
| `key` | 字符串键名 |
| `type` | `gguf_type`（UINT32/STRING/FLOAT/ARRAY/...） |
| `data` | 值（union） |

### 3.2 `gguf_tensor_info`

| 字段 | 说明 |
|------|------|
| `t` | `ggml_tensor` 元数据（name, ne, type） |
| `offset` | 在 data blob 中的字节偏移 |

### 3.3 `gguf_context`

| 字段 | 说明 |
|------|------|
| `kv[]` | metadata 键值对 |
| `info[]` | tensor 信息 |
| `alignment` | 数据对齐（默认 32，可被 KV 覆盖） |
| `data` | 映射/加载的权重 blob |

---

## 4. 关键 API

| API | 说明 |
|-----|------|
| `gguf_init_from_file(path, params)` | 从文件加载 |
| `gguf_init_from_callback(cb, params)` | 流式/自定义 IO |
| `gguf_free(ctx)` | 释放 |
| `gguf_get_val_str/i/f` | 读 KV 值 |
| `gguf_get_arr_data` | 读 KV 数组 |
| `gguf_get_tensor_name/offset/data` | 读 tensor |
| `gguf_get_n_tensors/n_kv` | 统计 |
| `gguf_write_to_file` | 写出 |

### 4.1 加载参数

```c
struct gguf_init_params {
    bool no_alloc;              // true: 只读 metadata，不分配 data
    struct ggml_context ** ctx; // 输出 ggml context
};
```

**llama.cpp 使用 `no_alloc=true`**：只建 tensor 元数据，真实 buffer 在 `create_tensor` 时按 Backend 分配。

---

## 5. 常见 Metadata KV

| Key | 类型 | 说明 |
|-----|------|------|
| `general.architecture` | string | 模型架构（llama, qwen3, ...） |
| `general.name` | string | 模型名 |
| `general.file_type` | uint32 | `ggml_ftype` 量化类型 |
| `general.quantization_version` | uint32 | 量化格式版本 |
| `general.alignment` | uint32 | 数据对齐字节 |
| `llama.context_length` | uint32 | 上下文长度 |
| `llama.embedding_length` | uint32 | 隐藏维度 |
| `llama.block_count` | uint32 | 层数 |
| `llama.attention.head_count` | uint32 | 注意力头数 |
| `tokenizer.ggml.model` | string | 分词器类型 |
| `tokenizer.ggml.tokens` | array | 词表 |
| `split.count` / `split.no` | uint16 | 分片信息 |

架构特定 KV 由 `llama-arch.cpp` 的 `LLM_KV_NAMES` 映射。

---

## 6. Tensor 命名规范

llama.cpp 通过 `LLM_TENSOR_INFOS` 映射 tensor 名到 `{op, layer}`：

| 模式 | 示例 | 层 |
|------|------|-----|
| Token embedding | `token_embd.weight` | INPUT |
| 层权重 | `blk.0.attn_q.weight` | REPEATING |
| 输出 | `output_norm.weight` | OUTPUT |

---

## 7. 分片（Multi-file GGUF）

大模型可拆为多个文件：

```
model-00001-of-00004.gguf  (split.no=0, split.count=4)
model-00002-of-00004.gguf  (split.no=1)
...
```

`llama_model_loader` 合并所有分片的 `weights_map`，统一加载。

---

## 8. 与旧格式对比

| 格式 | Magic | 说明 |
|------|-------|------|
| GGUF | `GGUF` | 当前标准，KV + tensor metadata |
| GGML binary | `ggml` (0x67676d6c) | 旧格式，已弃用 |

---

## 9. Python 工具

| 工具 | 路径 | 用途 |
|------|------|------|
| `gguf-py` | llama.cpp/gguf-py | Python 读写 GGUF |
| `convert_hf_to_gguf.py` | llama.cpp | HF -> GGUF 转换 |

---

## 10. 非显而易见细节

1. **`no_alloc=true`**：tensor `data` 指向文件 mmap 区域，不占用 RAM
2. **`general.alignment`**：覆盖默认 32 字节对齐，影响 offset 计算
3. **KV 类型错误会 abort**：`gguf_get_val_*` 类型不匹配时直接终止
4. **v3 vs v2**：v3 是当前版本，llama.cpp 只支持 GGUF
5. **ftype 推断**：若无 `general.file_type`，loader 统计 tensor 类型取众数

---

## 11. 读写示例

```python
# Python (gguf-py)
import gguf
reader = gguf.GGUFReader("model.gguf")
print(reader.fields["general.architecture"].parts[0])
```

```c
// C
struct gguf_init_params params = { .no_alloc = true, .ctx = &ctx };
struct gguf_context * meta = gguf_init_from_file("model.gguf", params);
const char * arch = gguf_get_val_str(meta, "general.architecture");
```

---

## 12. 相关文档

- [06-quantization.md](./06-quantization.md) - tensor type 与量化
- [13-llama-cpp-integration.md](./13-llama-cpp-integration.md) - llama_model_loader 解析流程
- `03-推理部署/llama.cppDoc/14-model-loader-deep-dive.md` - 加载器详解
