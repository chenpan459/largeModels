# 10 - Tools、Examples 与测试

## tools/ 目录

### 数据工具

| 脚本 | 说明 |
|------|------|
| `preprocess_data.py` | 文本/jsonl → Megatron `.bin/.idx` |
| `merge_datasets.py` | 合并多个 indexed 数据集 |
| `preprocess_mmdata.py` | 多模态数据预处理 |
| `preprocess_data_nmt.py` | 机器翻译格式 |
| `build_sequences_per_dataset.py` | 序列统计 |

### Checkpoint 工具

| 脚本/目录 | 说明 |
|-----------|------|
| `checkpoint/convert.py` | 格式转换总入口 |
| `checkpoint/loader_core.py` | 加载 Core checkpoint |
| `checkpoint/saver_core.py` | 保存 Core checkpoint |
| `checkpoint/loader_llava.py` | 多模态权重 |
| `checkpoint/hybrid_conversion.py` | Hybrid 模型转换 |
| `checkpoint/checkpoint_inspector.py` | 检查分片与 metadata |
| `checkpoint/dist_checkpoint_io.py` | 分布式 IO 辅助 |

### 推理与评测

| 脚本 | 说明 |
|------|------|
| `run_text_generation_server.py` | HTTP 文本生成 |
| `run_hybrid_text_generation_server.py` | Hybrid 模型服务 |
| `run_dynamic_text_generation_server.py` | 动态 batch |
| `run_inference_performance_test.py` | 推理性能测试 |
| `text_generation_cli.py` | 命令行交互 |

### 分析与开发

| 脚本 | 说明 |
|------|------|
| `report_theoretical_memory.py` | 理论显存占用 |
| `linter.py` | 代码检查 |
| `autoformat.sh` | ruff/black/isort 格式化 |
| `copyright.sh` | 版权头检查 |

## examples/ 目录

按 **模型 / 任务** 组织，典型内容：

- GPT / Llama 预训练 shell 脚本
- DeepSeek、Qwen、Mixtral **MoE** recipe
- T5、BERT 预训练
- Multimodal / LLaVA
- **RL** environments（`examples/rl/`）

**学习建议**：找与目标模型最接近的 example，复制 CLI 参数到本仓库实验环境，对照 [03-parallelism.md](./03-parallelism.md) 调整 TP/PP/EP。

## tests/

### 单元测试

`tests/unit_tests/` — 按模块划分：

| 目录 | 覆盖 |
|------|------|
| `transformer/` | attention、mlp、moe、mla、mtp |
| `transformer/moe/` | dispatcher、router、layer |
| `training/` | train_step、checkpoint、models |
| `distributed/` | DDP、FSDP |

示例：

```bash
pytest tests/unit_tests/transformer/test_attention.py -v
pytest tests/unit_tests/transformer/moe/test_moe_layer.py -v
```

### Functional tests

- Recipe YAML + golden values
- CI 在 NGC 容器 + GPU runner 上运行
- 仓库 `skills/mcore-testing/SKILL.md`（`.agents/skills/`）描述：
  - recipe 结构
  - golden value 更新流程
  - marker 过滤

## docs/ 与 skills/

| 资源 | 说明 |
|------|------|
| `docs/` | 与 https://docs.nvidia.com/megatron-core/ 同步的源 |
| `.agents/skills/mcore-testing` | 测试指南 |
| `.agents/skills/mcore-run-on-slurm` | SLURM 提交 |
| `.agents/skills/mcore-build-and-dependency` | 容器与 uv 依赖 |

## 开发工作流

```bash
cd /home/cp/work2/largeModels/06-Research/Megatron-LM

# 安装
uv pip install -e .
# 内存不足: MAX_JOBS=4 uv pip install -e .

# 格式化
./tools/autoformat.sh

# 单测
pytest tests/unit_tests/transformer/test_attention.py

# Mock 数据快速跑通（见 09-pretrain-gpt.md）
torchrun --nproc_per_node=1 pretrain_gpt.py --mock-data ...
```

## CI 架构（摘要）

- GitHub Actions + 内部 GitLab CI
- NGC PyTorch 容器（`docker/.ngc_version.dev`）
- Functional tests 需 GPU；MR 需 draft PR + CODEOWNERS

## 与本仓库 sync

根目录 `sync-repos.sh` 可更新 Megatron-LM；文档在 `Megatron-LMDoc/` 独立维护。

## 推荐阅读顺序

1. `tools/preprocess_data.py` — 数据格式
2. `examples/` 中最接近目标的 recipe
3. `tests/unit_tests/transformer/` — API 契约
4. 官方 Quickstart：https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html
