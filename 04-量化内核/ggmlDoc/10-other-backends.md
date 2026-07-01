# 10 - 其他 Backend

## 1. 概览

除 CPU/CUDA/Metal/Vulkan 外，GGML 还支持多种专用 Backend：

| Backend | 目录 | 平台 | CMake |
|---------|------|------|-------|
| SYCL | `ggml-sycl/` | Intel/跨厂商 GPU | `GGML_SYCL` |
| HIP | 共用 `ggml-cuda/` | AMD GPU | `GGML_HIP` |
| MUSA | 共用 `ggml-cuda/` | 摩尔线程 | `GGML_MUSA` |
| OpenCL | `ggml-opencl/` | 通用 GPU | `GGML_OPENCL` |
| RPC | `ggml-rpc/` | 远程 GPU | `GGML_RPC` |
| WebGPU | `ggml-webgpu/` | 浏览器 | `GGML_WEBGPU` |
| Hexagon | `ggml-hexagon/` | 高通 NPU | `GGML_HEXAGON` |
| CANN | `ggml-cann/` | 华为 Ascend | `GGML_CANN` |
| OpenVINO | `ggml-openvino/` | Intel 推理 | `GGML_OPENVINO` |
| BLAS | `ggml-blas/` | BLAS 加速 | `GGML_BLAS` |
| ZenDNN | `ggml-zendnn/` | AMD CPU | `GGML_ZENDNN` |
| zDNN | `ggml-zdnn/` | IBM Z | `GGML_ZDNN` |
| VirtGPU | `ggml-virtgpu/` | 虚拟 GPU | — |

---

## 2. SYCL（Intel GPU）

- 目录：`src/ggml-sycl/`
- 支持 Intel Arc、Data Center GPU
- 部分 kernel 从 CUDA 移植
- CMake：`GGML_SYCL=ON`，需 oneAPI

---

## 3. HIP / MUSA（AMD / 摩尔线程）

- **共用 CUDA 源码**：`ggml-cuda/` 通过 HIPify 或 MUSA 工具链编译
- CMake：`GGML_HIP=ON` 或 `GGML_MUSA=ON`
- API 与 CUDA Backend 一致，注册名不同

---

## 4. RPC Backend

远程 GPU 推理：

```
本地 llama.cpp
    |
    v
ggml-rpc backend -> TCP/网络
    |
    v
远程机器 ggml-rpc server (GPU)
```

用途：本地 CPU 机器调用远程 GPU 服务器推理。

CMake：`GGML_RPC=ON`

---

## 5. WebGPU

- 目录：`src/ggml-webgpu/`
- WGSL shader（`wgsl-shaders/*.wgsl`）
- 目标：浏览器/WASM 环境
- CMake：`GGML_WEBGPU=ON`

---

## 6. Hexagon（高通 NPU）

- 目录：`src/ggml-hexagon/`
- 高通骁龙 NPU 加速
- CMake：`GGML_HEXAGON=ON`
- 需 QNN SDK

---

## 7. CANN（华为 Ascend）

- 目录：`src/ggml-cann/`
- 华为昇腾 NPU
- CMake：`GGML_CANN=ON`

---

## 8. BLAS

- 目录：`src/ggml-blas/`
- 调用 OpenBLAS/MKL/Apple Accelerate
- 加速 F32 `MUL_MAT`
- CMake：`GGML_BLAS=ON`

---

## 9. Backend 动态加载

`GGML_BACKEND_DL=ON` 时，Backend 编译为独立 `.so`：

```
libggml-cpu.so
libggml-cuda.so
libggml-metal.so
    |
    v
ggml_backend_load_all()  # 扫描可执行文件目录
```

好处：主程序不链接所有 Backend，按需加载。

---

## 10. 选择建议

| 硬件 | 推荐 Backend |
|------|-------------|
| NVIDIA GPU | CUDA |
| Apple Silicon | Metal |
| AMD GPU | HIP (ROCm) |
| Intel GPU | SYCL 或 Vulkan |
| 通用 GPU | Vulkan |
| 仅 CPU | CPU (+ BLAS 可选) |
| 远程 GPU | RPC |
| 华为 Ascend | CANN |
| 高通手机 | Hexagon |

---

## 11. 相关文档

- [04-backend-scheduler.md](./04-backend-scheduler.md) - 多 Backend 调度
- [09-backend-gpu.md](./09-backend-gpu.md) - 主流 GPU Backend
- [11-build-system.md](./11-build-system.md) - 全部 CMake 选项
