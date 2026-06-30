# 14 - 模型加载器深度解析 (llama-model-loader)

## 1. 模块职责

`llama-model-loader.cpp`（约 1,704 行）是 llama.cpp 从 GGUF 文件加载模型的核心入口。它负责：

- 解析 GGUF 文件头与 metadata KV 对
- 识别模型架构 (`general.architecture`)
- 建立全局 tensor 索引 (`weights_map`)
- 按 backend buffer 类型分组创建 ggml context
- 支持 mmap / direct I/O / 异步 GPU 上传
- 处理多分片 (split) GGUF 文件

**头文件**: `src/llama-model-loader.h`  
**调用方**: 各 `llama_model_xxx::load_arch_tensors()` 通过 `create_tensor()` 创建权重

---

## 2. 核心数据结构

### 2.1 `llama_model_loader`

| 字段 | 类型 | 说明 |
|------|------|------|
| `metadata` | `gguf_context*` | GGUF 元数据上下文 |
| `weights_map` | `map<string, llama_tensor_weight>` | 全局 tensor 名 -> 权重描述 |
| `files` | `vector<unique_ptr<llama_file>>` | 分片文件句柄 |
| `contexts` | `vector<ggml_context*>` | 每分片一个 metadata context |
| `ctx_map` | `map<buft, ggml_context*>` | 按 buffer 类型分组的权重分配 context |
| `arch_name` | `string` | GGUF 中 `general.architecture` 字符串 |
| `llm_kv` | `LLM_KV` | 类型安全的 KV 键访问器 |
| `ftype` | `llama_ftype` | 量化类型（从 tensor 类型推断或 KV 读取） |
| `use_mmap` | `bool` | 是否内存映射 |
| `use_direct_io` | `bool` | 是否 direct I/O（与 mmap 互斥） |
| `check_tensors` | `bool` | 加载后校验 tensor 数据 |
| `no_alloc` | `bool` | 仅解析 metadata，不分配权重 |
| `mappings` | `vector<llama_mmap*>` | mmap 区域 |
| `kv_overrides` | `map<string, llama_model_kv_override>` | 用户 KV 覆盖 |
| `tensor_buft_overrides` | `const llama_model_tensor_buft_override*` | per-tensor buffer 覆盖 |

### 2.2 `llama_tensor_weight`

| 字段 | 说明 |
|------|------|
| `idx` | 所属分片文件索引 |
| `offs` | 在分片文件内的字节偏移 |
| `tensor` | 指向 ggml metadata tensor（仅 shape/type，无 data） |

### 2.3 Tensor 创建标志

| 标志 | 含义 |
|------|------|
| `TENSOR_NOT_REQUIRED` | 可选 tensor，缺失不报错 |
| `TENSOR_DUPLICATED` | 允许同名 tensor 复用（如 token_embd 兼作 output） |
| `TENSOR_SKIP` | 跳过创建 |
| `TENSOR_SKIP_IF_VIRTUAL` | 虚拟 tensor 时跳过 |

---

## 3. 加载流程

```
llama_model_loader 构造
  |
  +-- gguf_init_from_file(fname, no_alloc=true)     // L547: 只建 metadata
  +-- get_key(LLM_KV_GENERAL_ARCHITECTURE)          // L553
  +-- llm_kv = LLM_KV(llm_arch_from_string(...))    // L554
  +-- 遍历 ggml_get_first_tensor -> weights_map      // L576-585
  +-- [可选] 多分片: split_count/split_no 校验       // L586-665
  +-- 推断 ftype（多数 tensor 类型）或读 KV         // L711-780
  +-- 打印全部 KV（调试，最多 40 字符截断）          // L782-809

llama_model_xxx::load_tensors()
  |
  +-- load_arch_hparams(ml)   // 读超参数
  +-- load_arch_tensors(ml)   // create_tensor() x N
  +-- done_getting_tensors()  // 校验必需 tensor 已创建
  +-- init_mappings()         // 建立 mmap
  +-- load_all_data()         // mmap 或 read + 异步 GPU 上传
```

### 3.1 架构识别（L553-554）

```cpp
get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
llm_kv = LLM_KV(llm_arch_from_string(arch_name));
```

`llm_arch_from_string()` 在 `llama-arch.cpp:850` 查表 `LLM_ARCH_NAMES`，将字符串（如 `"llama"`、`"qwen3"`）映射为 `llm_arch` 枚举。

### 3.2 多分片加载（L586-665）

GGUF 分片通过 metadata 键 `split.count` / `split.no` 标识：

- 主文件必须是 `split.no == 0`
- 可通过 `llama_get_list_splits()` 自动生成 sibling 路径
- 所有分片的 tensor 合并到统一 `weights_map`

### 3.3 `create_tensor()`（L1047）

核心逻辑：

1. 从 `weights_map` 查找 tensor 元数据
2. 通过 `llm_tensor_info_for()` 获取 `{op, layer}` 元信息
3. 根据 layer 类型（INPUT / OUTPUT / REPEATING）选择 `buft_list`
4. 支持 per-tensor `tensor_buft_overrides`
5. 在 `ctx_map[buft]` 中分配 ggml tensor + backend buffer
6. bias 后缀 tensor 自动改 op 为 `GGML_OP_ADD` / `GGML_OP_ADD_ID`

### 3.4 `load_all_data()`（L1408）

两种路径：

| 模式 | 行为 |
|------|------|
| **mmap** | `init_mappings()` 映射文件到内存，tensor.data 直接指向映射区域 |
| **read + upload** | 4 个 64MB pinned buffer + event 流水线异步上传到 GPU |

非 mmap 模式下，CPU 读文件与 GPU 上传可并行，减少加载延迟。

---

## 4. 关键 API

| 函数 | 行号 | 职责 |
|------|------|------|
| `llama_model_loader()` | 512 | 构造函数，GGUF 解析入口 |
| `get_arch_name()` | 823 | 返回架构字符串 |
| `get_arch()` | 827 | 返回 `llm_arch` 枚举 |
| `get_tensor_meta()` | 848 | 仅获取 metadata tensor |
| `require_tensor_meta()` | 856 | 必需 tensor，缺失抛异常 |
| `create_tensor()` | 1047 | 创建带 buffer 的实际权重 |
| `create_tensor_as_view()` | 1289 | 视图 tensor（共享 buffer） |
| `done_getting_tensors()` | 1317 | 校验所有必需 tensor |
| `init_mappings()` | 1335 | 建立 mmap |
| `load_all_data()` | 1408 | 批量加载权重数据 |
| `get_key<T>()` | 模板 | 类型安全 KV 读取 |
| `get_arr<T>()` | 模板 | KV 数组读取 |

### 4.1 模板 KV 读取

```cpp
template<typename T>
bool get_key(enum llm_kv kid, T & result, bool required = true);

template<typename T>
bool get_key_or_arr(enum llm_kv kid, std::array<T, N> & result, uint32_t n, bool required);
```

`LLM_KV` 类提供类型安全的 KV 键名生成，避免硬编码字符串。

---

## 5. 与架构注册的关系

加载器只负责 **识别** 架构，模型类 **实例化** 在 `llama-model.cpp:38-304` 的 `llama_model_mapping()`：

```
GGUF arch string
  -> llm_arch_from_string()     // llama-arch.cpp
  -> llama_model_create(arch)   // llama-model.cpp
  -> switch(arch) new llama_model_xxx(params)
  -> load_arch_hparams(ml)
  -> load_arch_tensors(ml)      // 调用 create_tensor()
```

新增架构需同步修改三处：`llm_arch` 枚举、`LLM_ARCH_NAMES`、`llama_model_mapping`。

---

## 6. 非 obvious 实现细节

### 6.1 `no_alloc=true` 两阶段加载

GGUF 初始化阶段只创建 tensor **元数据**（shape、type、name），不分配权重内存。真实 buffer 在 `create_tensor()` 时按目标 backend 分配。这样可以在知道完整 tensor 列表后再统一规划 buffer 布局。

### 6.2 mmap 与 direct I/O 互斥（L559-571）

二者不可同时启用。若 direct I/O 可用则禁用 mmap；否则降级为 mmap 并重新打开文件。

### 6.3 `TENSOR_DUPLICATED`

`token_embd.weight` 可同时作为 `output.weight`（weight tying）。第二次 `create_tensor` 时传入 `TENSOR_DUPLICATED` 标志，复用已有 buffer 但可能使用不同 ggml op。

### 6.4 KV override 不应用于 dump

构造时写入的 `kv_overrides` 在 debug dump metadata 时不生效（L782 注释），避免调试输出与真实加载参数不一致。

### 6.5 ftype 推断（L711-780）

优先从 KV `general.file_type` 读取；若不存在，统计所有 tensor 的量化类型，取出现次数最多的类型作为 ftype。

---

## 7. 环境变量

| 变量 | 作用 |
|------|------|
| `LLAMA_TRACE` | >0 时打印分片加载等详细日志 |

---

## 8. 错误处理

| 场景 | 行为 |
|------|------|
| GGUF 文件损坏 | 构造时 `throw std::runtime_error` |
| 重复 tensor 名 | L579-580 抛异常 |
| 分片 idx != 0 作为主文件 | L595-596 抛异常 |
| 必需 tensor 缺失 | `done_getting_tensors()` 抛异常 |
| tensor 数据校验失败 | `check_tensors` 模式下报错 |

---

## 9. 扩展指南

| 需求 | 修改位置 |
|------|----------|
| 新 GGUF KV 键 | `llama-arch.h` 添加 `llm_kv` 枚举 + `LLM_KV_NAMES` |
| 新 tensor 命名规范 | `LLM_TENSOR_INFOS` 添加 `{op, layer}` |
| per-arch 加载逻辑 | `src/models/xxx.cpp` 的 `load_arch_hparams/tensors` |
| 新 buffer 分配策略 | `create_tensor()` 中的 buft 选择逻辑 |

---

## 10. 相关文档

- [03-src-core.md](./03-src-core.md) - libllama 核心概览
- [04-models.md](./04-models.md) - 模型架构层
- [09-conversion-gguf.md](./09-conversion-gguf.md) - GGUF 格式
- [16-kv-cache-memory.md](./16-kv-cache-memory.md) - Memory 工厂（依赖 arch 识别结果）
