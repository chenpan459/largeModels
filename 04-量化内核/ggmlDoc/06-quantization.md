# 06 - 量化系统

## 1. 模块概述

| 文件 | 职责 |
|------|------|
| `src/ggml-common.h` | 量化 block 结构体（跨 CPU/CUDA/Metal/Vulkan 共享） |
| `src/ggml-quants.h` | 量化 API 声明 |
| `src/ggml-quants.c` | 参考实现（~5,591 行） |
| `src/ggml-cpu/quants.c` + `arch/*/quants.c` | CPU SIMD vec_dot |
| `src/ggml-cpu/repack.cpp` | 运行时 layout 转换 |
| `src/ggml-cuda/mmq.cu`, `mmvq.cu` | GPU 量化矩阵乘 |

---

## 2. 量化 Block 格式（`ggml-common.h`）

### 2.1 标准 Block（QK=32）

| Block | 结构 | 约 bit/weight |
|-------|------|---------------|
| `block_q1_0` | QK=128, `half d` + 1bit packed | ~1 |
| `block_q4_0` | `half d` + 16 byte nibbles | ~4.5 |
| `block_q4_1` | `d` + `m` + nibbles | ~5.0 |
| `block_q5_0/1` | 5bit + delta | ~5.5 |
| `block_q8_0` | `half d` + int8[32] | ~8.5 |
| `block_mxfp4` | E8M0 scale | OCP MX |
| `block_nvfp4` | 4×UE4M3 sub-scales | NVIDIA FP4 |

```c
#define QK4_0 32
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;
```

### 2.2 K-Quants（Super-Block QK_K=256）

| Block | bit 级 | 特点 |
|-------|--------|------|
| `block_q2_K` | 2-bit | scale/min 层次 |
| `block_q3_K` | 3-bit | |
| `block_q4_K` | 4-bit | scale 分 high/low；**Q4_K_M 常用** |
| `block_q5_K` | 5-bit | |
| `block_q6_K` | 6-bit | |
| `block_q8_K` | 8-bit | |

256 元素 super-block 内共享 scale → 更高压缩率 + 更好质量。

### 2.3 Importance 量化（IQ 系列）

| 类型 | bit | 特点 |
|------|-----|------|
| IQ1_S, IQ1_M | ~1 | 极低 bit |
| IQ2_XXS, IQ2_XS, IQ2_S | ~2 | codebook |
| IQ3_XXS, IQ3_S | ~3 | |
| IQ4_NL, IQ4_XS | ~4 | non-linear |

需 `iq2xs_init_impl()` / `ggml_quantize_init()` 惰性初始化 codebook。

### 2.4 Ternary / 新格式

| 类型 | 说明 |
|------|------|
| TQ1_0, TQ2_0 | 三值量化 |
| MXFP4, NVFP4 | 浮点 4-bit |

### 2.5 SIMD 宏 QI / QR

`ggml-common.h` L100-167：

- `QI4_0`：一个 SIMD 块内 4-bit 元素数
- `QR4_0`：行重复因子

影响 CPU kernel tile 大小与循环展开。

---

## 3. 量化 API

### 3.1 参考实现（`ggml-quants.h`）

| 函数 | 说明 |
|------|------|
| `quantize_row_q4_0_ref` | F32 → Q4_0 |
| `dequantize_row_q4_0` | Q4_0 → F32 |
| `quantize_q4_K(...)` | 批量 K-quant |
| `ggml_vec_dot_q4_0_q8_0` | **推理核心点积** |

### 3.2 统一入口（`ggml.c`）

```c
size_t ggml_quantize_chunk(
    enum ggml_type type,
    const float * src, void * dst,
    int64_t start, int64_t n, int64_t nrows,
    const float * imatrix);   // per-column 重要性权重

bool ggml_quantize_requires_imatrix(enum ggml_type type);  // IQ 系列强制
void ggml_quantize_init(enum ggml_type type);
```

### 3.3 imatrix（重要性矩阵）

- 由 `llama-imatrix` 工具从校准数据生成
- API 参数名 `quant_weights` / `imatrix`：per-column 权重
- K-quants 中 `make_qp_quants()` 用权重加权选码（`ggml-quants.c` L1018+）
- **IQ 系列强制要求** imatrix，否则质量极差

### 3.4 type_traits（`ggml.c`）

每种 `ggml_type`：`blck_size`, `type_size`, `to_float`, `from_float_ref`。

---

## 4. vec_dot 体系（CPU 推理核心）

声明：`src/ggml-cpu/quants.h` L40-67

模式：**量化权重 × Q8 激活**

| vec_dot 函数 | 权重类型 |
|--------------|----------|
| `ggml_vec_dot_q4_0_q8_0` | Q4_0 |
| `ggml_vec_dot_q4_1_q8_1` | Q4_1 |
| `ggml_vec_dot_q4_K_q8_K` | Q4_K |
| `ggml_vec_dot_q5_K_q8_K` | Q5_K |
| `ggml_vec_dot_q6_K_q8_K` | Q6_K |
| `ggml_vec_dot_iq2_xxs_q8_K` | IQ2_XXS |
| … | 30+ 组合 |

实现分层：

```
ggml-quants.c          → generic 参考
ggml-cpu/arch/x86/quants.c   → AVX/AVX512/VNNI
ggml-cpu/arch/arm/quants.c   → NEON/dotprod/i8mm
ggml-cpu/arch/riscv/quants.c → RVV
ggml-cpu/quants.c      → 调度到 arch
```

`ggml_compute_forward_mul_mat`（`ggml-cpu.c` L1245+）按 `src0->type` 选择 vec_dot 函数指针。

---

## 5. Repack（运行时 Layout 转换）

| 文件 | 说明 |
|------|------|
| `ggml-cpu/repack.cpp` | 主逻辑 |
| `ggml-cpu/arch/x86/repack.cpp` | x86 Q4_0→Q4_X_X |
| `ggml-cpu/arch/arm/repack.cpp` | ARM |
| `ggml-cpu/spacemit/repack.cpp` | SpacemiT RVV |

**Repack buffer type**：`ggml_backend_cpu_repack_buffer_type()`（`repack.h`）

- CMake：`GGML_CPU_REPACK=ON`（默认）
- 加载权重到 repack buffer 时自动转换 layout
- matmul 更快，加载时多一步 CPU 开销

Extra buffer types（`ggml-cpu.cpp` L42-66）：

- repack（`GGML_CPU_REPACK`）
- KleidiAI（`GGML_CPU_KLEIDIAI`）
- SpacemiT（RISC-V）

---

## 6. 推理量化路径

```
权重 (Q4_K in GGUF)
    |
    v
[可选] repack buffer → Q4_X layout
    |
    v
激活在线量化为 Q8_0（mul_mat 内部）
    |
    v
ggml_mul_mat(qweight, q8_act)
    |
    +-- CPU: vec_dot_* (SIMD)
    +-- CUDA: mmq.cu / mmvq.cu
    +-- Metal: ggml-metal.metal quant kernel
    +-- Vulkan: mul_mmq.comp / mul_mat_vec_*.comp
    |
    v
F32 输出 logits / hidden
```

CUDA 路由（`ggml-cuda.cu` L2541+）：

```
vec case → mmvq (quant) / mmvf (float)
batched quant → mmq
float batched → cublas
```

环境变量：`GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS`

---

## 7. 各 Backend 量化实现

| Backend | 文件 | 技术 |
|---------|------|------|
| CPU 参考 | `ggml-quants.c` | 纯 C；llama-quantize 用 |
| CPU SIMD | `arch/x86/quants.c` | AVX512 VNNI |
| CPU ARM | `arch/arm/quants.c` | NEON dotprod |
| CPU AMX | `amx/mmq.cpp` | Intel AMX int8 |
| CUDA MMQ | `mmq.cu` | `ggml_cuda_mul_mat_q` |
| CUDA MMVQ | `mmvq.cu` | mat-vec |
| Vulkan | `mul_mmq.comp` 等 | shader-gen 25 类型变体 |
| Metal | `ggml-metal.metal` | MSL quant kernels |

Flash Attention 量化：`GGML_CUDA_FA_ALL_QUANTS` 启用全部类型 FA kernel。

---

## 8. `ggml_ftype` 与 GGUF

| ftype | 含义 |
|-------|------|
| `GGML_FTYPE_ALL_F32` | 全 F32 |
| `GGML_FTYPE_MOSTLY_Q4_0` | 大部分 Q4_0 |
| `GGML_FTYPE_MOSTLY_Q4_K_M` | **生产常用** |
| `GGML_FTYPE_MOSTLY_Q8_0` | 高精度量化 |

GGUF metadata：`general.file_type`、`general.quantization_version`（对应 `GGML_QNT_VERSION=2`）。

---

## 9. llama-quantize 流程

```
llama-quantize in.gguf out.gguf Q4_K_M [--imatrix file]
    |
    +-- gguf_init_from_file(in)
    +-- 对每个 tensor:
    |     dequantize → float
    |     quantize_q4_K(..., imatrix?) → quantized
    +-- gguf_write_to_file(out)
```

使用 `ggml-quants.c` 参考 quantize，**非** Backend kernel。

---

## 10. 非显而易见细节

1. **`GGML_QNT_VERSION=2`**：block 布局变更需 bump
2. **enum 末尾追加**：保证旧 GGUF 可读
3. **激活 Q8**：权重 Q4 + 激活 Q8 是默认推理数学
4. **已移除类型**：Q4_2/Q4_3 enum 位保留
5. **dequant 仅工具/debug**：推理 hot path 是 vec_dot，不全量 dequant

---

## 11. 扩展指南

| 需求 | 位置 |
|------|------|
| 新量化类型 | `ggml-common.h` + `ggml_type` enum + `ggml-quants.c` |
| 新 CPU kernel | `ggml-cpu/arch/*/quants.c` + `quants.h` 声明 |
| 新 GPU kernel | `ggml-cuda/*.cu` + `supports_op` |
|  Vulkan shader | `vulkan-shaders/` + `vulkan-shaders-gen.cpp` |
| 质量调优 | imatrix + IQ/K-quants 混合 |

## 相关文档

- [07-gguf-format.md](./07-gguf-format.md)
- [08-backend-cpu.md](./08-backend-cpu.md)
- [09-backend-gpu.md](./09-backend-gpu.md)
- [vllmDoc/15-quantization-catalog.md](../../03-推理部署/vllmDoc/15-quantization-catalog.md) — HF 量化对照
