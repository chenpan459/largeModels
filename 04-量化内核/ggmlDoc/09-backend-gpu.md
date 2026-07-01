# 09 - GPU Backend（CUDA / Metal / Vulkan）

## 1. 总览

| Backend | 主文件 | 行数(约) | 平台 |
|---------|--------|----------|------|
| CUDA | `ggml-cuda/ggml-cuda.cu` | 5,721 | NVIDIA |
| Metal | `ggml-metal/ggml-metal.cpp` + `.metal` | 950 + 10,754 | Apple |
| Vulkan | `ggml-vulkan/ggml-vulkan.cpp` | 18,696 | 跨平台 GPU |

HIP（AMD）和 MUSA（摩尔线程）共用 CUDA 目录，CMake 别名编译。

---

## 2. CUDA Backend

### 2.1 目录结构（~80 `.cu` 文件）

| 文件 | 职责 |
|------|------|
| `ggml-cuda.cu` | 设备管理、buffer、supports_op、注册 |
| `mmq.cu` | 量化矩阵乘（MMQ） |
| `mmvq.cu` | 量化 mat-vec |
| `fattn.cu`, `fattn-*.cu` | Flash Attention 系列 |
| `rope.cu` | RoPE |
| `norm.cu` | RMSNorm/LayerNorm |
| `softmax.cu` | Softmax |
| `cpy.cu` | 拷贝 |
| `allreduce.cu` | 多卡 NCCL |
| `quantize.cu` | 量化工具 |

### 2.2 注册与初始化

```c
ggml_backend_reg_t reg = ggml_backend_cuda_reg();  // L5650
ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, gpu_id);
ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
```

### 2.3 关键特性

| 特性 | CMake/环境 | 说明 |
|------|-----------|------|
| MMQ | 默认 | 量化 matmul 主力 |
| cuBLAS fallback | `GGML_CUDA_FORCE_CUBLAS` | 强制 cuBLAS |
| MMQ 强制 | `GGML_CUDA_FORCE_MMQ` | 强制自定义 kernel |
| Flash Attention | `GGML_CUDA_FA` | FA CUDA kernel |
| CUDA Graph | `GGML_CUDA_GRAPHS` | 减少 launch 开销 |
| Multi-GPU | split buffer | 仅 `MUL_MAT` tensor parallel |
| Pinned memory | 默认 | 加速 H2D 拷贝 |

### 2.4 `supports_op` 限制

- tensor 必须在对应 GPU 的 buffer 上
- split buffer 模式仅支持 `MUL_MAT`
- 部分 op 需特定 CUDA 版本或 GPU 架构

---

## 3. Metal Backend

### 3.1 文件结构

| 文件 | 行数 | 职责 |
|------|------|------|
| `ggml-metal.cpp` | 950 | Backend 注册 |
| `ggml-metal-device.cpp/.m` | 2086/1900 | MTLDevice/Buffer/Queue |
| `ggml-metal-ops.cpp` | 4,633 | 算子 dispatch -> shader |
| `ggml-metal.metal` | 10,754 | 全部 GPU kernel |

### 3.2 特性

| 特性 | 说明 |
|------|------|
| 默认开启 | macOS `GGML_METAL_DEFAULT ON` |
| Embed library | `GGML_METAL_EMBED_LIBRARY` shader 嵌入二进制 |
| offload 阈值 | `offload_op`: batch < 阈值不 offload |
| Buffer 类型 | Shared/Private/Mapped |
| Apple Silicon | M1/M2/M3/M4 优化 |

### 3.3 `offload_op`

Metal 对小 batch 的 `MUL_MAT` 可能不 offload 到 GPU（CPU 更快），通过 `ggml_backend_dev_offload_op` 判断。

---

## 4. Vulkan Backend

### 4.1 结构

单体大文件 `ggml-vulkan.cpp`（18,696 行）+ 100+ compute shader：

```
vulkan-shaders/
├── mul_mat*.comp
├── flash_attn*.comp
├── rope.comp
├── norm.comp
├── softmax.comp
└── ...
```

`vulkan-shaders-gen.cpp`：编译期生成 shader 变体（量化类型 x tile size）。

### 4.2 特性

| 特性 | 说明 |
|------|------|
| 跨平台 | Linux/Windows/Android GPU |
| 运行时 specialization | 量化类型 x tile size 组合 |
| BDA | Buffer Device Address 大 tensor |
| 禁用 | `GGML_DISABLE_VULKAN=1` 环境变量 |

---

## 5. Backend 对比

| 维度 | CUDA | Metal | Vulkan |
|------|------|-------|--------|
| 默认 | OFF | macOS ON | OFF |
| 量化 matmul | MMQ/MMVQ | metal shader | comp shader |
| Flash Attn | fattn*.cu | metal | flash_attn*.comp |
| 多卡 | NCCL split | 有限 | 有限 |
| 代码组织 | 按 op 分文件 | cpp + 大 .metal | 单体 cpp |
| 成熟度 | 最高 | Apple 优 | 快速发展 |

---

## 6. GPU 内存管理

```
ggml_backend_buft_alloc_buffer(buft, size)
    |
    v
ggml_backend_buffer (GPU VRAM)
    |
    +-- USAGE_WEIGHTS: 模型权重（持久）
    +-- USAGE_COMPUTE: 中间激活（图内复用）
    |
    v
ggml_backend_tensor_copy(src, dst)  # 跨 Backend 拷贝
```

llama.cpp：`n_gpu_layers` 控制前 N 层权重分配到 GPU buffer。

---

## 7. llama.cpp GPU 参数

| 参数 | 效果 |
|------|------|
| `-ngl N` / `n_gpu_layers` | 前 N 层 offload 到 GPU |
| `--tensor-split` | 多卡负载比例 |
| `-fa on` | Flash Attention |
| `--no-mmap` | 禁用 mmap，直接读入 GPU |

---

## 8. 相关文档

- [04-backend-scheduler.md](./04-backend-scheduler.md) - GPU 在 sched 中的分配
- [06-quantization.md](./06-quantization.md) - GPU 量化 kernel
- [10-other-backends.md](./10-other-backends.md) - SYCL/HIP 等
- [11-build-system.md](./11-build-system.md) - GPU CMake 选项
