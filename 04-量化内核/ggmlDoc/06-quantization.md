# 06 - 量化系统

## 1. 模块概述

| 文件 | 职责 |
|------|------|
| `src/ggml-common.h` | 量化 block 结构体定义 |
| `src/ggml-quants.h` | 量化 API 声明 |
| `src/ggml-quants.c` | 参考实现（~5,591 行） |
| `src/ggml-cpu/quants.c` | CPU SIMD 加速 |
| `src/ggml-cpu/arch/{x86,arm,riscv}/quants.c` | 架构专用 intrinsics |
| `src/ggml-cpu/repack.cpp` | 运行时 Q4_0 -> Q4_X repack |
| `src/ggml-cuda/mmq.cu`, `mmvq.cu` | GPU 量化矩阵乘 |

---

## 2. 量化 Block 格式（`ggml-common.h`）

### 2.1 标准 Block

| Block | 元素数 | 结构 | 约 bit/weight |
|-------|--------|------|---------------|
| `block_q4_0` | 32 | `half delta` + 16 byte 4bit | ~4.5 |
| `block_q4_1` | 32 | `half delta` + `half min` + 4bit | ~5.0 |
| `block_q5_0` | 32 | 5bit + delta | ~5.5 |
| `block_q8_0` | 32 | `half delta` + int8[32] | ~8.5 |

```c
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // 4-bit quants (packed)
} block_q4_0;
```

### 2.2 K-Quants（Super-Block）

| Block | Super-block 大小 | 特点 |
|-------|------------------|------|
| `block_q2_K` | QK_K = 256 | 2-bit + scale/min 层次 |
| `block_q3_K` | 256 | 3-bit |
| `block_q4_K` | 256 | 4-bit，scale 分 high/low |
| `block_q5_K` | 256 | 5-bit |
| `block_q6_K` | 256 | 6-bit |
| `block_q8_K` | 256 | 8-bit |

K-quants 在 256 元素 super-block 内共享 scale，压缩率更高，质量更好。

### 2.3 Importance 量化（IQ 系列）

| 类型 | bit | 特点 |
|------|-----|------|
| IQ1_S, IQ1_M | ~1 | 极低 bit |
| IQ2_XXS, IQ2_XS, IQ2_S | ~2 | 需 codebook |
| IQ3_XXS, IQ3_S | ~3 | |
| IQ4_NL, IQ4_XS | ~4 | non-linear |

需 `iq2xs_init_impl()` 等初始化 codebook。

### 2.4 新格式

| 类型 | 说明 |
|------|------|
| MXFP4 | OCP MX 浮点 4-bit |
| NVFP4 | NVIDIA FP4 |
| TQ1_0, TQ2_0 | Ternary 量化 |

---

## 3. 量化 API（`ggml-quants.h`）

### 3.1 参考实现

| 函数 | 说明 |
|------|------|
| `quantize_row_q4_0_ref` | F32 -> Q4_0 |
| `dequantize_row_q4_0` | Q4_0 -> F32 |
| `quantize_row_q4_1_ref` | F32 -> Q4_1 |
| `ggml_vec_dot_q4_0_q8_0` | 量化点积（推理核心） |

### 3.2 批量量化（K-quants）

```c
size_t quantize_q4_K(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
                     const float * quant_weights);  // imatrix 可选
```

`quant_weights`（imatrix）：per-column 重要性权重，AWQ 风格量化时使用。

### 3.3 类型 traits（`ggml.c` type_traits[]）

每种 `ggml_type` 关联：

- `blck_size`：block 元素数
- `type_size`：单个 block 字节数
- `to_float`：dequant 函数指针
- `from_float_ref`：quant 函数指针

---

## 4. 推理中的量化路径

```
权重 (Q4_K in GGUF)
    |
    v
[可选] repack: Q4_0 -> Q4_X_X (GGML_CPU_REPACK)
    |
    v
ggml_mul_mat(qweight, q8_activations)
    |
    +-- CPU: arch/quants.c SIMD dot product
    +-- CUDA: mmq.cu / mmvq.cu
    +-- Metal: ggml-metal.metal quant kernel
    |
    v
F32 输出
```

**关键**：激活值通常量化为 Q8_0，与 Q4 权重做点积（`vec_dot_q4_0_q8_0`）。

---

## 5. 各 Backend 量化实现

| Backend | 文件 | 技术 |
|---------|------|------|
| CPU 参考 | `ggml-quants.c` | 纯 C，量化工具用 |
| CPU SIMD | `ggml-cpu/arch/x86/quants.c` | AVX/AVX512 VNNI |
| CPU ARM | `ggml-cpu/arch/arm/quants.c` | NEON, dotprod |
| CPU Repack | `ggml-cpu/repack.cpp` | 运行时 layout 转换 |
| CPU AMX | `ggml-cpu/amx/mmq.cpp` | Intel AMX int8 |
| CUDA MMQ | `ggml-cuda/mmq.cu` | 量化矩阵乘 |
| CUDA MMVQ | `ggml-cuda/mmvq.cu` | 量化 mat-vec |
| Metal | `ggml-metal.metal` | GPU shader |

---

## 6. `ggml_ftype` 与 GGUF

| ftype | 含义 |
|-------|------|
| `GGML_FTYPE_ALL_F32` | 全 F32 |
| `GGML_FTYPE_MOSTLY_Q4_0` | 大部分 Q4_0 |
| `GGML_FTYPE_MOSTLY_Q4_K_M` | 大部分 Q4_K_M（常用） |
| `GGML_FTYPE_MOSTLY_Q8_0` | 大部分 Q8_0 |

GGUF metadata `general.file_type` 存储 ftype 枚举值。

---

## 7. 非显而易见细节

1. **`GGML_QNT_VERSION=2`**：量化格式版本号，变更 block 布局需 bump
2. **enum 末尾追加**：新 `ggml_type` 只在 enum 末尾添加，保证旧 GGUF 可读
3. **imatrix**：`llama-imatrix` 生成 per-column 重要性，传给 `quantize_*` 提升 IQ 系列质量
4. **已移除类型**：Q4_2/Q4_3 等 enum 位保留但注释 "removed"
5. **Repack 性能**：`GGML_CPU_REPACK` 将 Q4_0 转为 CPU 友好 layout，matmul 更快但加载时多一步

---

## 8. llama-quantize 流程

```
llama-quantize in.gguf out.gguf Q4_K_M
    |
    +-- gguf_init_from_file(in)
    +-- 对每个 tensor:
    |     dequantize -> float
    |     quantize_q4_K(..., imatrix?) -> quantized
    +-- gguf_write_to_file(out)
```

使用 `ggml-quants.c` 的参考 quantize 函数，非 Backend kernel。

---

## 9. 扩展指南

| 需求 | 位置 |
|------|------|
| 新量化类型 | `ggml-common.h` block + `ggml_type` enum + `ggml-quants.c` |
| 新 CPU kernel | `ggml-cpu/arch/*/quants.c` |
| 新 GPU kernel | `ggml-cuda/` 新 .cu 文件 + `supports_op` |
| 量化质量调优 | imatrix + IQ 系列 + K-quants 混合 |

---

## 10. 相关文档

- [07-gguf-format.md](./07-gguf-format.md) - 量化类型在 GGUF 中的存储
- [08-backend-cpu.md](./08-backend-cpu.md) - CPU repack 与 SIMD
- [09-backend-gpu.md](./09-backend-gpu.md) - CUDA MMQ/MMVQ
