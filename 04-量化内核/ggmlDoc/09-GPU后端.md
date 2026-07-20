# 09 - GPU Backend（CUDA / Metal / Vulkan）

## 1. 总览

| Backend | 主文件 | 行数(约) | 平台 |
|---------|--------|----------|------|
| CUDA | `ggml-cuda/ggml-cuda.cu` | 5,721 + 65 .cu | NVIDIA |
| HIP / MUSA | 共用 `ggml-cuda/` | — | AMD / 摩尔线程 |
| Metal | `ggml-metal/` 多文件 | ~20k+ | Apple |
| Vulkan | `ggml-vulkan/ggml-vulkan.cpp` | 18,696 | 跨平台 GPU |

HIP/MUSA 通过 CMake 别名编译同一源码树，注册名不同。

---

## 2. CUDA Backend

### 2.1 .cu 文件分类（65 个）

| 类别 | 文件 |
|------|------|
| 矩阵乘 | `mmq.cu`, `mmvq.cu`, `mmvf.cu`, `mmf.cu`, `mmid.cu` |
| Flash Attn | `fattn.cu`, `fattn-tile.cu`, `fattn-wmma-f16.cu` |
| 归一化/激活 | `norm.cu`, `softmax.cu`, `unary.cu` |
| 位置编码 | `rope.cu` |
| MoE | `topk-moe.cu`, `mmid.cu` |
| 状态空间 | `ssm-scan.cu`, `ssm-conv.cu`, `gated_delta_net.cu`, `gla.cu` |
| 卷积 | `conv2d*.cu`, `im2col.cu`, `pool2d.cu` |
| RWKV | `wkv.cu` |
| 拷贝/量化 | `cpy.cu`, `convert.cu`, `quantize.cu` |
| 多卡 | `allreduce.cu`（NCCL，`GGML_CUDA_NCCL`） |
| 训练 | `opt-step*.cu` |

### 2.2 上下文结构

`ggml-cuda.cu`：

- `ggml_backend_cuda_context`：device、cublas handle、memory pool
- `ggml_backend_cuda_buffer_context`：VRAM buffer（L626+）
- `tensor->extra`：CUDA tensor 私有元数据

### 2.3 MUL_MAT 路由（L2541-2621）

```
n_dims==2 且 vec case:
  quant → mmvq (ggml_cuda_mul_mat_vec_q)
  float → mmvf

else quant:
  → mmq (ggml_cuda_mul_mat_q)

else float batched:
  → cublas (ggml_cuda_mul_mat_batched_cublas)
```

强制开关：

- `GGML_CUDA_FORCE_MMQ=1`
- `GGML_CUDA_FORCE_CUBLAS=1`

### 2.4 Flash Attention

| 选项 | 说明 |
|------|------|
| `GGML_CUDA_FA=ON` | 默认启用 |
| `GGML_CUDA_FA_ALL_QUANTS` | 全部量化类型 FA kernel |
| `fattn-tile.cu` | tile 变体 |
| `fattn-wmma-f16.cu` | WMMA FP16 |

### 2.5 其他特性

| 特性 | 说明 |
|------|------|
| CUDA Graph | `GGML_CUDA_GRAPHS`（llama.cpp only） |
| VMM | `GGML_CUDA_NO_VMM` 禁用 |
| Pinned host | 默认启用，加速 H2D |
| Multi-GPU split | 仅 `MUL_MAT` / `MUL_MAT_ID` |
| NCCL | `GGML_CUDA_NCCL` 多卡 allreduce |

### 2.6 supports_op 限制（L5054+）

- tensor 必须在对应 GPU 的 CUDA buffer
- split buffer **仅** `MUL_MAT` / `MUL_MAT_ID`
- 大量 op 要求 contiguous
- UNARY/GLU 按子类型白名单
- 部分 op 需特定 SM 版本

---

## 3. Metal Backend（多文件架构）

Metal 已从单文件重构为分层结构：

| 文件 | 职责 |
|------|------|
| `ggml-metal.cpp` | Backend 注册、buffer 接口 |
| `ggml-metal-device.cpp/.m` | MTLDevice/Buffer/Queue |
| `ggml-metal-context.m` | 渲染/计算上下文 |
| `ggml-metal-ops.cpp` | 算子 dispatch → pipeline（L201 supports_op） |
| `ggml-metal-common.cpp` | 公共工具 |
| `ggml-metal.metal` | 全部 MSL kernel（~10,754 行） |

### 特性

| 特性 | 说明 |
|------|------|
| macOS 默认 | `GGML_METAL_DEFAULT ON` |
| Embed shader | `GGML_METAL_EMBED_LIBRARY` 嵌入二进制 |
| offload 阈值 | `ggml_backend_metal_device_offload_op()`：小 batch 不 offload |
| Buffer 类型 | Shared / Private / Mapped |
| 多设备 | `GGML_METAL_DEVICES` 环境变量 |

接口：

- `ggml_metal_device_supports_op()`（`ggml-metal-device.m` L1051）

详见 [15-Metal与Vulkan深度解析.md](./15-Metal与Vulkan深度解析.md)。

---

## 4. Vulkan Backend

### 主实现

`ggml-vulkan.cpp`（~18,696 行）：device、buffer、pipeline、dispatch。

### Shader 目录

`src/ggml-vulkan/vulkan-shaders/`（132+ 文件）：

| 类别 | 文件示例 |
|------|----------|
| 矩阵乘 | `mul_mm*.comp`, `mul_mmq.comp` |
| Mat-vec | `mul_mat_vec_q4_k.comp`, `mul_mat_vec_iq2_xxs.comp` |
| Flash Attn | `flash_attn*.comp`, `flash_attn_cm2.comp` |
| Dequant | `dequant_q4_k.comp`, ... |
| RoPE | `rope_neox.comp`, `rope_multi.comp`, `rope_norm.comp` |
| 共享 | `types.glsl`, `utils.glsl`, `rope_funcs.glsl` |

### Shader 生成器

`vulkan-shaders-gen.cpp`：

- 用 `glslc` 编译 `.comp` → SPIR-V
- `type_names[]`（L49-75）：25 种量化/浮点类型 Cartesian 积
- `MatMulIdType`：DEFAULT/SUBGROUP 等变体
- 输出 `ggml-vulkan-shaders.hpp` 嵌入二进制
- `ASYNCIO_CONCURRENCY=64` 并行编译

运行时：`GGML_DISABLE_VULKAN=1` 可禁用。

详见 [15-Metal与Vulkan深度解析.md](./15-Metal与Vulkan深度解析.md)。

---

## 5. HIP / MUSA

- **共用** `ggml-cuda/` 源码
- CMake：`GGML_HIP=ON` 或 `GGML_MUSA=ON`
- HIPify 或 MUSA 工具链编译
- API 与 CUDA 一致，注册名 `HIP` / `MUSA`

---

## 6. 跨 Backend 对照

| 能力 | CUDA | Metal | Vulkan |
|------|------|-------|--------|
| 量化 MM | mmq.cu | .metal shader | mul_mmq.comp |
| Flash Attn | fattn*.cu | .metal | flash_attn*.comp |
| RoPE | rope.cu | .metal | rope_*.comp |
| Shader 语言 | CUDA C++ | MSL | GLSL → SPIR-V |
| 编译时 | nvcc | metallib | glslc + gen |

---

## 7. 性能调优

| 目标 | CUDA | Metal | Vulkan |
|------|------|-------|--------|
| 量化推理 | Q4_K + MMQ | Q4_K 模型 | Q4_K 模型 |
| Flash Attn | `-fa on` + FA | `-fa on` | `-fa on` |
| 小 batch | MMVQ | offload 阈值 | mat-vec shader |
| 多 GPU | tensor-split | 有限 | 有限 |
| 调试 | `--verbose` | `GGML_METAL_DEBUG` | `GGML_VULKAN_DEBUG` |

---

## 相关文档

- [06-量化系统.md](./06-量化系统.md)
- [15-Metal与Vulkan深度解析.md](./15-Metal与Vulkan深度解析.md)
- [11-构建系统.md](./11-构建系统.md)
