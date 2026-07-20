# 08 - CPU Backend

## 1. 模块概述

CPU Backend 是 **兜底 Backend**：`supports_op` 返回 true  for 全部算子，sched 中优先级最低。

| 文件/目录 | 行数(约) | 职责 |
|-----------|----------|------|
| `ggml-cpu.cpp` | 703 | Backend 注册、extra buffer types |
| `ggml-cpu.c` | 3,840 | **`ggml_graph_compute`** 主循环 |
| `ops.cpp` | 11,514 | 各 `ggml_op` CPU 实现 |
| `quants.c` | 1,288 | SIMD vec_dot 调度 |
| `repack.cpp` | 4,836 | 权重 repack |
| `arch/x86/quants.c` | 6,596 | AVX/AVX512/VNNI |
| `arch/arm/quants.c` | 3,970 | NEON/dotprod/i8mm |
| `arch/riscv/quants.c` | — | RVV |
| `amx/mmq.cpp` | 2,511 | Intel AMX INT8 |
| `llamafile/sgemm.cpp` | 4,052 | 可选 F32 GEMM |
| `kleidiai/` | 1,523 | ARM KleidiAI |
| `spacemit/` | — | SpacemiT RISC-V 优化 |
| `vec.cpp` | — | F32/F16/BF16 vec_dot |
| `traits.h` | — | supports_op 抽象 |

## 2. 执行路径

```
ggml_backend_cpu_graph_compute(backend, cgraph)
    ↓
ggml_graph_plan(cgraph, n_threads)
    ↓
ggml_graph_compute(cplan)                    // ggml-cpu.c L3308
    ↓
for each node in cgraph->nodes:
    ggml_compute_forward(params, tensor)     // L1702, 大 switch
    ↓
ops.cpp 中 ggml_compute_forward_* 实现
```

量化 matmul 路径：

```
ggml_compute_forward_mul_mat (ggml-cpu.c L1245+)
  → 激活量化为 Q8
  → 按 src0->type 选择 vec_dot 函数指针
  → arch/*/quants.c SIMD 实现
```

## 3. 线程模型

| 机制 | CMake / API | 说明 |
|------|-------------|------|
| OpenMP | `GGML_OPENMP=ON` | 节点内 `#pragma omp parallel` |
| Threadpool | `ggml-threading.cpp` | `ggml_threadpool_new`、affinity |
| 手动 | `cplan.n_threads` | llama `-t N` |
| NUMA | `ggml_numa_init()` | 多 socket 绑定 |

Threadpool 支持 cpumask、disposable threadpool（短生命周期任务）。

## 4. Arch 目录结构

```
ggml-cpu/arch/
├── x86/       quants.c, repack.cpp, cpu-feats.cpp
├── arm/       quants.c, repack.cpp, cpu-feats.cpp
├── riscv/     quants.c, repack.cpp, cpu-feats.cpp
├── powerpc/   quants.c, cpu-feats.cpp
├── s390/      quants.c
├── loongarch/ quants.c
└── wasm/      quants.c
```

`cpu-feats.cpp`：运行时 ISA 检测；配合 `GGML_CPU_ALL_VARIANTS` + `GGML_BACKEND_DL` 选择最优 CPU 变体（haswell、skylake、apple_m4、riscv64_v 等）。

## 5. SIMD / ISA 支持

### x86

| ISA | CMake | 用途 |
|-----|-------|------|
| AVX / AVX2 | `GGML_AVX` / `GGML_AVX2` | 256-bit SIMD |
| AVX512 | `GGML_AVX512` | 512-bit |
| AVX512_VNNI | — | int8 dot product |
| AMX INT8 | `GGML_AMX_INT8` | `amx/mmq.cpp` |
| AMX BF16 | `GGML_AMX_BF16` | BF16 矩阵乘 |

AMX 条件：`__AMX_INT8__ && __AVX512VNNI__`（`amx/mmq.cpp` L35+）。

### ARM

| ISA | 说明 |
|-----|------|
| NEON | 128-bit SIMD |
| dotprod | int8 点积 |
| i8mm | int8 矩阵乘 |
| SVE | 可变长向量 |
| KleidiAI | `GGML_CPU_KLEIDIAI` 优化库 + 专用 buffer type |

### RISC-V

| ISA | 说明 |
|-----|------|
| RVV | 向量扩展 |
| ZVFH | FP16 向量 |
| SpacemiT | `spacemit/` 专用 repack + quants |

## 6. Extra Buffer Types

`ggml_backend_cpu_get_extra_buffer_types()`（`ggml-cpu.cpp` L42-66）：

| Buffer Type | 触发 | 效果 |
|-------------|------|------|
| repack | `GGML_CPU_REPACK=ON` | Q4_0→Q4_X layout，matmul 加速 |
| KleidiAI | `GGML_CPU_KLEIDIAI` | ARM 优化 buffer |
| SpacemiT | RISC-V SpacemiT | RVV 优化 repack |

加载权重到 repack buffer 时在 `init_tensor` 中自动 repack。

## 7. 量化 kernel 层次

```
ggml-quants.c           参考 dequant/quant（工具用）
    ↓
ggml-cpu/quants.c       调度层
    ↓
arch/x86/quants.c       AVX512 VNNI vec_dot_q4_K_q8_K
arch/arm/quants.c       NEON dotprod
arch/riscv/quants.c     RVV
```

完整 vec_dot 列表见 `ggml-cpu/quants.h` L40-67（30+ 组合）。

## 8. F32 矩阵乘

| 路径 | 条件 |
|------|------|
| llamafile SGEMM | `GGML_LLAMAFILE=ON` |
| BLAS | `GGML_BLAS=ON`（OpenBLAS/MKL/Accelerate） |
| 纯 C | ops.cpp 朴素实现 |

## 9. supports_op

CPU Backend **支持全部** `ggml_op`（包括训练 `OPT_STEP_*`、新 op `GATED_DELTA_NET` 等）。

sched 中 CPU 为最低优先级；仅当 GPU 不支持或权重在 CPU buffer 时使用。

## 10. 性能调优

| 目标 | 方法 |
|------|------|
| 量化推理 | Q4_K 模型 + `GGML_CPU_REPACK` |
| x86 吞吐 | AVX512_VNNI + AMX（Xeon） |
| Apple | KleidiAI buffer |
| 线程 | `-t` 设为物理核数；NUMA 绑定 |
| 调试 | `GGML_DEBUG=1` 打印 node 执行 |

## 11. 扩展新 op

1. `ggml.h` 添加 `GGML_OP_XXX`
2. `ggml.c` 添加工厂函数
3. `ops.cpp` 实现 `ggml_compute_forward_xxx`
4. `ggml-cpu.c` 的 switch 注册
5. `tests/test-backend-ops.cpp` 添加测试

GPU Backend 可选实现；CPU 必须实现。

## 相关文档

- [06-量化系统.md](./06-量化系统.md)
- [11-构建系统.md](./11-构建系统.md)
- [14-ggml-opt与线程系统.md](./14-ggml-opt与线程系统.md)
