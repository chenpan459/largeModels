# 13 - 测试与 CI

## 1. 模块概述

llama.cpp 拥有完善的测试体系和 CI/CD 流水线，覆盖单元测试、集成测试和多平台构建验证。

涉及目录：
- `tests/` - C++ 单元测试
- `tools/server/tests/` - Server Python 集成测试
- `gguf-py/tests/` - Python GGUF 库测试
- `.github/workflows/` - CI/CD 工作流
- `ci/` - CI 脚本

## 2. C++ 单元测试 (tests/)

构建控制: `LLAMA_BUILD_TESTS=ON` (默认)

### 2.1 测试清单

| 测试文件 | 说明 |
|----------|------|
| `test-tokenizer-0.cpp` | 基础分词器 |
| `test-tokenizer-1-bpe.cpp` | BPE 分词器 |
| `test-tokenizer-1-spm.cpp` | SentencePiece 分词器 |
| `test-tokenizer-random.py` | 随机分词测试 |
| `test-grammar-parser.cpp` | GBNF 语法解析 |
| `test-grammar-*.cpp` | Grammar 集成测试 |
| `test-sampling.cpp` | 采样器 |
| `test-chat.cpp` | Chat 模板 |
| `test-chat-peg-parser.cpp` | PEG Chat parser |
| `test-chat-auto-parser.cpp` | Auto parser |
| `test-jinja.cpp` | Jinja 模板引擎 |
| `test-json-partial.cpp` | 部分 JSON 解析 |
| `test-json-schema-to-grammar.cpp` | JSON Schema 转 Grammar |
| `test-regex-partial.cpp` | 部分正则匹配 |
| `test-peg-parser.cpp` | PEG 解析器 |
| `test-gguf.cpp` | GGUF 读写 |
| `test-gguf-model-data.cpp` | GGUF 模型数据 |
| `test-quantize-fns.cpp` | 量化函数 |
| `test-quantize-perf.cpp` | 量化性能 |
| `test-opt.cpp` | 优化器 |
| `test-barrier.cpp` | 线程屏障 |
| `test-thread-safety.cpp` | 线程安全 |
| `test-autorelease.cpp` | 自动释放 |
| `test-log.cpp` | 日志系统 |
| `test-recurrent-state-rollback.cpp` | 循环状态回滚 |
| `test-col2im-1d.cpp` | col2im 操作 |
| `test-c.c` | C API 兼容性 |

### 2.2 运行测试

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release
cd build && ctest --output-on-failure
```

## 3. Server 集成测试

**路径**: `tools/server/tests/`

Python pytest 测试套件，需要运行中的 server 实例。

### 3.1 测试模块

| 文件 | 说明 |
|------|------|
| `test_basic.py` | 基本 API 功能 |
| `test_chat_completion.py` | Chat 补全 |
| `test_completion.py` | 文本补全 |
| `test_embedding.py` | 嵌入向量 |
| `test_tokenize.py` | 分词 API |
| `test_template.py` | Chat 模板 |
| `test_tool_call.py` | Function calling |
| `test_rerank.py` | 重排序 |
| `test_vision_api.py` | 多模态视觉 |
| `test_speculative.py` | 投机解码 |
| `test_lora.py` | LoRA 适配 |
| `test_infill.py` | 代码填充 |
| `test_slot_save.py` | Slot 持久化 |
| `test_ctx_shift.py` | Context 滑动 |
| `test_kv_keep_only_active.py` | KV cache 优化 |
| `test_ignore_eos.py` | EOS 忽略 |
| `test_sleep.py` | Sleep/Wake |
| `test_security.py` | 安全测试 |
| `test_proxy.py` | 代理 |
| `test_router.py` | 多模型路由 |
| `test_compat_oai_responses.py` | OpenAI Responses 兼容 |
| `test_compat_anthropic.py` | Anthropic 兼容 |
| `test_compat_gcp.py` | GCP 兼容 |

### 3.2 运行

```bash
# 启动 server
llama-server -m model.gguf --port 8080 &

# 运行测试
cd tools/server/tests
pip install -r requirements.txt
./tests.sh
# 或
pytest unit/ -v
```

## 4. Python 测试

### 4.1 gguf-py 测试

```bash
cd gguf-py
pip install -e .
pytest tests/
```

### 4.2 转换脚本测试

```bash
# 分词器一致性
python tests/test-tokenizer-0.py

# Jinja 模板
python scripts/jinja/jinja-tester.py
```

## 5. CI/CD 工作流

**路径**: `.github/workflows/`

### 5.1 构建工作流

| 工作流 | 说明 |
|--------|------|
| `build-cpu.yml` | CPU 构建 (Linux/macOS/Windows) |
| `build-cuda-ubuntu.yml` | NVIDIA CUDA 构建 |
| `build-apple.yml` | Apple Silicon (Metal) |
| `build-android.yml` | Android NDK |
| `build-riscv.yml` | RISC-V |
| `build-sycl.yml` | Intel SYCL |
| `build-opencl.yml` | OpenCL |
| `build-webgpu.yml` | WebGPU |
| `build-cann.yml` | 华为 CANN |
| `build-msys.yml` | Windows MSYS2 |
| `build-cross.yml` | 交叉编译 |
| `build-ibm.yml` | IBM s390x |
| `build-openvino.yml` | Intel OpenVINO |
| `build-rpc.yml` | RPC 后端 |
| `build-snapdragon.yml` | 高通骁龙 |
| `build-sanitize.yml` | Sanitizer 构建 |
| `build-self-hosted.yml` | 自托管 runner |
| `build-virtgpu.yml` | 虚拟 GPU |

### 5.2 质量工作流

| 工作流 | 说明 |
|--------|------|
| `server.yml` | Server 构建 + 测试 |
| `server-sanitize.yml` | Server Sanitizer |
| `server-self-hosted.yml` | Server 自托管测试 |
| `code-style.yml` | 代码风格检查 |
| `python-lint.yml` | Python lint |
| `python-type-check.yml` | Python 类型检查 |
| `editorconfig.yml` | EditorConfig |
| `check-vendor.yml` | 第三方依赖检查 |
| `hip-quality-check.yml` | AMD HIP 质量 |
| `pre-tokenizer-hashes.yml` | 分词器哈希 |

### 5.3 发布工作流

| 工作流 | 说明 |
|--------|------|
| `release.yml` | GitHub Release |
| `docker.yml` | Docker 镜像 |
| `winget.yml` | Windows Winget 包 |
| `gguf-publish.yml` | GGUF 发布 |
| `ui-publish.yml` | Web UI 发布 |
| `ui-build.yml` | UI 构建 |

## 6. CI 脚本

**路径**: `ci/`

| 文件 | 说明 |
|------|------|
| `run.sh` | 主 CI 运行脚本 |
| `README.md` | CI 说明 |
| `README-MUSA.md` | MUSA CI 说明 |

## 7. 代码质量

### 7.1 代码风格

- C/C++: `.clang-format` + `.clang-tidy`
- Python: `.flake8` + `pyrightconfig.json`
- EditorConfig: `.editorconfig`

### 7.2 Pre-commit 检查

CI 工作流 `code-style.yml` 验证：
- clang-format 格式
- EditorConfig 合规
- Python flake8

## 8. 基准测试

**路径**: `benches/`

| 目录 | 说明 |
|------|------|
| `mac-m2-ultra/` | Apple M2 Ultra 基准 |
| `dgx-spark/` | NVIDIA DGX 基准 |
| `nemotron/` | Nemotron 基准 |

**脚本**:

| 脚本 | 说明 |
|------|------|
| `scripts/bench-models.sh` | 模型基准 |
| `scripts/server-bench.py` | Server 基准 |
| `scripts/tool_bench.sh` | 工具基准 |
| `tools/server/bench/bench.py` | Server HTTP 基准 |

## 9. 本地开发测试建议

```bash
# 1. 构建 + 单元测试
cmake -B build -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure

# 2. 快速功能验证
./build/bin/llama-cli -m model.gguf -p "Hello" -n 16

# 3. Server 测试
./build/bin/llama-server -m model.gguf --port 8080 &
cd tools/server/tests && pytest unit/test_basic.py -v

# 4. 量化测试
./build/bin/llama-quantize model-f16.gguf /tmp/test-q4.gguf Q4_0
./build/bin/llama-cli -m /tmp/test-q4.gguf -p "test" -n 8

# 5. Sanitizer 构建
cmake -B build-asan -DLLAMA_SANITIZE_ADDRESS=ON
cmake --build build-asan -j$(nproc)
```

## 10. 测试模型

测试使用的模型通常从 HuggingFace 下载 small models：

- `ggml-org/models` - 官方测试模型
- CI 中使用 `cmake/download-models.cmake` 自动下载
- 测试脚本 `tests/get-model.cpp` 提供模型路径解析
