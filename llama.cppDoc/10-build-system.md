# 10 - 构建系统

## 1. 概述

llama.cpp 使用 **CMake 3.14+** 构建系统，支持多平台、多后端交叉编译。根 `CMakeLists.txt` 协调 ggml、libllama、common、tools、examples 和 tests 的构建。

## 2. 快速构建

```bash
# 基本 CPU 构建
cmake -B build
cmake --build build --config Release -j$(nproc)

# NVIDIA GPU
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Apple Silicon (Metal 自动启用)
cmake -B build
cmake --build build --config Release -j$(nproc)

# 安装
cmake --install build --prefix /usr/local
```

## 3. 构建产物

| 产物 | 类型 | 说明 |
|------|------|------|
| `libggml.so` | 共享库 | GGML 计算引擎 |
| `libllama.so` | 共享库 | 核心推理库 |
| `libllama-common.so` | 共享库 | 公共工具库 |
| `llama-cli` | 可执行 | 命令行推理 |
| `llama-server` | 可执行 | HTTP 服务 |
| `llama-quantize` | 可执行 | 量化工具 |
| `llama-bench` | 可执行 | 性能测试 |

输出目录: `build/bin/`

## 4. 主要 CMake 选项

### 4.1 llama.cpp 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `LLAMA_BUILD_COMMON` | ON | 构建 common 库 |
| `LLAMA_BUILD_TESTS` | ON | 构建测试 |
| `LLAMA_BUILD_TOOLS` | ON | 构建工具 |
| `LLAMA_BUILD_EXAMPLES` | ON | 构建示例 |
| `LLAMA_BUILD_SERVER` | ON | 构建 server |
| `LLAMA_BUILD_APP` | ON | 构建统一二进制 |
| `LLAMA_BUILD_UI` | ON | 构建 Web UI |
| `LLAMA_USE_PREBUILT_UI` | ON | 使用预构建 UI |
| `LLAMA_OPENSSL` | ON | HTTPS 支持 |
| `LLAMA_FATAL_WARNINGS` | OFF | -Werror |
| `LLAMA_SANITIZE_ADDRESS` | OFF | ASan |
| `LLAMA_SANITIZE_THREAD` | OFF | TSan |
| `LLAMA_LLGUIDANCE` | OFF | LLGuidance 结构化输出 |
| `LLAMA_USE_SYSTEM_GGML` | OFF | 使用系统 libggml |
| `BUILD_SHARED_LIBS` | ON | 动态库 (MinGW/WASM 除外) |

### 4.2 GGML 后端选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `GGML_CPU` | ON | CPU 后端 |
| `GGML_CUDA` | OFF | NVIDIA CUDA |
| `GGML_METAL` | 平台 | Apple Metal |
| `GGML_VULKAN` | OFF | Vulkan GPU |
| `GGML_SYCL` | OFF | Intel SYCL |
| `GGML_HIP` | OFF | AMD ROCm |
| `GGML_MUSA` | OFF | 摩尔线程 MUSA |
| `GGML_OPENCL` | OFF | OpenCL |
| `GGML_OPENVINO` | OFF | Intel OpenVINO |
| `GGML_CANN` | OFF | 华为 CANN |
| `GGML_RPC` | OFF | 远程 GPU RPC |
| `GGML_WEBGPU` | OFF | WebGPU |
| `GGML_HEXAGON` | OFF | 高通 Hexagon |
| `GGML_BLAS` | 平台 | BLAS 加速 |
| `GGML_NATIVE` | ON | 本机 CPU 优化 |
| `GGML_LLAMAFILE` | ON | llamafile SGEMM |
| `GGML_CUDA_GRAPHS` | ON | CUDA Graph |
| `GGML_BACKEND_DL` | OFF | 后端动态加载 |
| `GGML_CPU_REPACK` | ON | 运行时 weight repack |

### 4.3 CPU SIMD 选项

| 选项 | 说明 |
|------|------|
| `GGML_AVX` / `GGML_AVX2` / `GGML_AVX512` | x86 SIMD |
| `GGML_AVX512_VNNI` / `GGML_AVX512_BF16` | x86 扩展 |
| `GGML_CPU_KLEIDIAI` | ARM KleidiAI |
| `GGML_RVV` / `GGML_RV_ZVFH` | RISC-V 向量 |

## 5. 平台特定构建

### 5.1 NVIDIA CUDA

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89"  # RTX 4090
cmake --build build --config Release -j$(nproc)
```

### 5.2 AMD ROCm (HIP)

```bash
cmake -B build -DGGML_HIP=ON
cmake --build build --config Release -j$(nproc)
```

### 5.3 Apple Metal

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(nproc)
```

### 5.4 Intel SYCL

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j$(nproc)
```

### 5.5 Android

```bash
cmake -B build-android \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28
cmake --build build-android --config Release
```

### 5.6 WebAssembly

```bash
emcmake cmake -B build-wasm \
    -DLLAMA_BUILD_SERVER=ON \
    -DGGML_WEBGPU=ON
emmake make -j$(nproc) -C build-wasm
```

## 6. CMake Presets

`CMakePresets.json` 提供预配置构建方案：

```bash
cmake --preset default
cmake --build --preset default
```

可用 presets 包括各平台的 Release/Debug 配置。

## 7. 交叉编译

`cmake/` 目录包含工具链文件：

| 文件 | 目标 |
|------|------|
| `arm64-linux-clang.cmake` | ARM64 Linux |
| `arm64-apple-clang.cmake` | Apple Silicon |
| `arm64-windows-llvm.cmake` | ARM64 Windows |
| `x64-windows-llvm.cmake` | x64 Windows |
| `riscv64-spacemit-linux-gnu-gcc.cmake` | RISC-V |

## 8. 构建信息

构建时自动生成版本信息：

```cpp
// common/build-info.cpp (from template)
#define LLAMA_BUILD_NUMBER  ...
#define LLAMA_COMMIT        "cf93b3b"
```

可通过 `llama-cli --version` 查看。

## 9. Docker 构建

```bash
# CPU
docker build -t llama-cpp:cpu -f .devops/cpu.Dockerfile .

# CUDA
docker build -t llama-cpp:cuda -f .devops/cuda.Dockerfile .

# 运行
docker run -v /models:/models llama-cpp:cpu -m /models/model.gguf
```

## 10. 依赖关系图

```
CMakeLists.txt (root)
    |
    +-- ggml/CMakeLists.txt
    |       +-- ggml-cpu/
    |       +-- ggml-cuda/ (if GGML_CUDA)
    |       +-- ggml-metal/ (if GGML_METAL)
    |       +-- ...
    |
    +-- src/CMakeLists.txt -> libllama
    |
    +-- common/CMakeLists.txt -> libllama-common
    |       +-- vendor/cpp-httplib
    |
    +-- tools/CMakeLists.txt
    |       +-- cli/, server/, quantize/, ...
    |
    +-- examples/CMakeLists.txt
    |
    +-- tests/CMakeLists.txt
    |
    +-- app/CMakeLists.txt (if LLAMA_BUILD_APP)
```

## 11. 常见问题

| 问题 | 解决方案 |
|------|----------|
| CUDA not found | 设置 `CUDA_PATH` 或 `-DCMAKE_CUDA_COMPILER` |
| Out of memory (编译) | 减少 `-j` 并行数 |
| Metal not available | 仅 macOS/iOS 支持 |
| 静态链接 | `-DBUILD_SHARED_LIBS=OFF` |
| 调试构建 | `-DCMAKE_BUILD_TYPE=Debug` |
| 禁用所有工具 | `-DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF` |

## 12. 参考文档

- 官方构建指南: `docs/build.md`
- Android: `docs/android.md`
- Docker: `docs/docker.md`
- 多 GPU: `docs/multi-gpu.md`
- 安装: `docs/install.md`
