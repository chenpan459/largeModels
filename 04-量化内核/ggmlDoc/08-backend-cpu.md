# 08 - CPU Backend

## 1. 模块概述

| 文件/目录 | 行数(约) | 职责 |
|-----------|----------|------|
| `src/ggml-cpu/ggml-cpu.cpp` | 703 | Backend 注册、入口 |
| `src/ggml-cpu/ggml-cpu.c` | 3,840 | **`ggml_graph_compute`** 主循环 |
| `src/ggml-cpu/ops.cpp` | 11,514 | 各 `ggml_op` CPU 实现 |
| `src/ggml-cpu/quants.c` | 1,288 | SIMD 量化 kernel |
| `src/ggml-cpu/repack.cpp` | 4,836 | 权重 repack |
| `src/ggml-cpu/arch/x86/quants.c` | 6,596 | AVX/AVX512 |
| `src/ggml-cpu/arch/arm/quants.c` | 3,970 | NEON/dotprod |
| `src/ggml-cpu/amx/mmq.cpp` | 2,511 | Intel AMX |
| `src/ggml-cpu/llamafile/sgemm.cpp` | 4,052 | 可选 fast GEMM |
| `src/ggml-cpu/kleidiai/` | 1,523 | ARM KleidiAI |

CPU Backend 是 **兜底 Backend**：支持全部算子，sched 中优先级最低。

---

## 2. 执行入口

```
ggml_backend_cpu_graph_compute(backend, cgraph)
    |
    v
ggml_graph_plan(cgraph, n_threads)   # 规划线程任务
    |
    v
ggml_graph_compute(cplan)            # ggml-cpu.c L3308
    |
    v
for each node in cgraph->nodes:
    dispatch -> ops.cpp 中对应函数
```

---

## 3. 线程模型

| 机制 | CMake 选项 | 说明 |
|------|-----------|------|
| OpenMP | `GGML_OPENMP=ON` | 默认，节点内并行 |
| 手动线程池 | `ggml_graph_plan.n_threads` | llama.cpp 设 `n_threads` |
| NUMA | `ggml_numa_init()` | 多 socket 绑定 |

---

## 4. SIMD 架构支持

### 4.1 x86

| ISA | CMake | 用途 |
|-----|-------|------|
| AVX | `GGML_AVX` | 基础 SIMD |
| AVX2 | `GGML_AVX2` | 256-bit |
| AVX512 | `GGML_AVX512` | 512-bit |
| AVX512_VNNI | — | int8 dot product |
| AMX | `GGML_AMX_*` | int8 矩阵乘 |

### 4.2 ARM

| ISA | 说明 |
|-----|------|
| NEON | 基础 128-bit SIMD |
| dotprod | int8 点积加速 |
| i8mm | int8 矩阵乘 |
| SVE | 可变长向量 |
| KleidiAI | `GGML_CPU_KLEIDIAI` 优化库 |

### 4.3 RISC-V

| ISA | 说明 |
|-----|------|
| RVV | 向量扩展 |
| ZVFH | FP16 向量 |

---

## 5. Repack（`GGML_CPU_REPACK`）

运行时将 Q4_0 权重转为 CPU 友好 layout（Q4_X_X）：

```
GGUF Q4_0 weights
    |
    v
repack.cpp (加载时或首次 matmul)
    |
    v
Q4_X_X layout (更适合 SIMD 访问)
    |
    v
arch/quants.c 加速 matmul
```

权衡：加载时间增加，推理 matmul 更快。

---

## 6. Llamafile SGEMM（`GGML_LLAMAFILE`）

`llamafile/sgemm.cpp`：多线程 SGEMM 实现，F16/F32 矩阵乘 fallback。

---

## 7. CPU 变体（`GGML_CPU_ALL_VARIANTS`）

构建多个 ISA 变体库（haswell, skylakex, icelake, apple_m4 等），运行时 `dlopen` 选最优：

```
GGML_BACKEND_DL=ON + GGML_CPU_ALL_VARIANTS=ON
    |
    v
libggml-cpu-haswell.so
libggml-cpu-skylakex.so
libggml-cpu-apple_m4.so
    |
    v
运行时检测 CPU -> 加载最优变体
```

---

## 8. `supports_op`

CPU Backend **支持全部算子**，作为 sched 的最后 fallback。当 GPU 不支持某 op 时，该节点落回 CPU。

---

## 9. 性能调优

| 参数/选项 | 效果 |
|-----------|------|
| `n_threads` | 并行度（llama.cpp `--threads`） |
| `GGML_NATIVE=ON` | 针对本机 CPU 编译优化 |
| `GGML_CPU_REPACK=ON` | 量化 matmul 加速 |
| `GGML_LLAMAFILE=ON` | F32 GEMM 加速 |
| `GGML_AMX_*` | Intel Sapphire Rapids+ AMX |

---

## 10. 相关文档

- [06-quantization.md](./06-quantization.md) - CPU 量化 kernel
- [09-backend-gpu.md](./09-backend-gpu.md) - GPU vs CPU 分工
- [11-build-system.md](./11-build-system.md) - CPU ISA CMake 选项
