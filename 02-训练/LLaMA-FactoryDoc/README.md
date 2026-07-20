# LLaMA Factory 项目文档

本目录包含对 `/home/cp/work2/largeModels/02-训练/LLaMA-Factory` 项目的结构化源码分析文档。

## 文档索引

### 环境搭建

| 文档 | 说明 |
|------|------|
| [00-环境搭建指南.md](./00-环境搭建指南.md) | **环境搭建**：Python 3.11、虚拟环境、安装、排错 |

### 概览与架构

| 文档 | 说明 |
|------|------|
| [01-LLaMA-Factory项目概览.md](./01-LLaMA-Factory项目概览.md) | 项目定位、功能特性、技术栈 |
| [02-架构设计.md](./02-架构设计.md) | 模块分层、依赖关系、v0/v1 双架构 |

### 入口与配置

| 文档 | 说明 |
|------|------|
| [03-CLI与入口.md](./03-CLI与入口.md) | CLI 命令、入口脚本、分布式启动 |
| [04-配置系统.md](./04-配置系统.md) | YAML 配置、hparams 参数体系 |

### 核心模块

| 文档 | 说明 |
|------|------|
| [05-训练流水线.md](./05-训练流水线.md) | 训练流程、各 stage 路由、回调 |
| [06-数据模块.md](./06-数据模块.md) | 数据集加载、模板、处理器 |
| [07-模型模块.md](./07-模型模块.md) | 模型加载、LoRA/量化、补丁 |
| [08-推理与Chat.md](./08-推理与Chat.md) | ChatModel、推理引擎、API 服务 |
| [09-Web-UI与LLaMA-Board.md](./09-Web-UI与LLaMA-Board.md) | LLaMA Board Gradio 界面 |

### 实践与参考

| 文档 | 说明 |
|------|------|
| [10-使用指南.md](./10-使用指南.md) | 训练、导出、部署示例 |
| [11-API参考.md](./11-API参考.md) | 关键函数与参数速查 |

## 项目路径

```
/home/cp/work2/largeModels/02-训练/LLaMA-Factory/
├── src/
│   ├── train.py              # 训练入口脚本
│   ├── api.py                # API 入口脚本
│   ├── webui.py              # Web UI 入口脚本
│   └── llamafactory/         # 核心库
│       ├── cli.py            # CLI 入口
│       ├── launcher.py       # 子命令分发
│       ├── train/            # 训练流水线
│       ├── data/             # 数据处理
│       ├── model/            # 模型加载
│       ├── hparams/          # 超参数
│       ├── chat/             # 推理引擎
│       ├── api/              # FastAPI 服务
│       ├── webui/            # Gradio 界面
│       ├── eval/             # 评测
│       └── v1/               # 实验性 v1 架构
├── examples/                 # YAML 配置示例
├── data/                     # 演示数据集 + dataset_info.json
├── requirements/             # 可选依赖
├── scripts/                  # 工具脚本
└── tests/                    # 单元测试
```

## 推荐阅读顺序

1. **首次安装**：00 → 验证 `llamafactory-cli version`
2. **快速上手**：01 → 10 → 04
3. **理解训练流程**：02 → 03 → 05 → 06 → 07
4. **部署推理**：08 → 10（API 部署部分）
5. **可视化操作**：09
6. **二次开发**：02 → 05 → 06 → 07 → 11

## 快速参考

```bash
# 安装
pip install -e .

# LoRA 微调
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# 启动 Web UI
llamafactory-cli webui

# 启动 OpenAI 兼容 API
llamafactory-cli api

# 合并 LoRA 并导出
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

## 上游项目

- 仓库: https://github.com/hiyouga/LLaMA-Factory
- 许可证: Apache 2.0
- 官方文档: https://llamafactory.readthedocs.io
