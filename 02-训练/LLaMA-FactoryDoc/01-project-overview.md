# 01 — 项目概览

## 项目定位

**LLaMA Factory** 是一个开源的大语言模型（LLM）与多模态模型统一微调框架。它用零代码或低代码方式，支持 100+ 模型的训练、评测、推理与导出，并提供 CLI、Gradio Web UI（LLaMA Board）和 OpenAI 兼容 API 三种交互方式。

项目被 Amazon SageMaker、NVIDIA RTX AI Toolkit、阿里云 PAI 等生产环境采用。

## 核心能力

| 能力 | 说明 |
|------|------|
| 多模型支持 | LLaMA、Qwen3、Qwen3-VL、DeepSeek、Gemma、GLM、Mistral、Mixtral-MoE 等 |
| 训练阶段 | 预训练 (pt)、监督微调 (sft)、奖励建模 (rm)、PPO、DPO、KTO、ORPO、SimPO |
| 微调方法 | 全量微调、Freeze、LoRA、OFT |
| 量化 | 2–8 bit QLoRA（bitsandbytes、GPTQ、AWQ、AQLM、HQQ、EETQ、FP8） |
| 多模态 | 图像、视频、音频理解与对话 |
| 分布式 | 多 GPU torchrun、DeepSpeed ZeRO、FSDP/FSDP2、Ray、Megatron-core |
| 推理后端 | HuggingFace、vLLM、SGLang、KTransformers |
| 实验监控 | LlamaBoard、TensorBoard、W&B、MLflow、SwanLab |

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python ≥ 3.11 |
| 深度学习 | PyTorch、Transformers、Accelerate、PEFT、TRL |
| 数据 | HuggingFace Datasets、ModelScope |
| Web UI | Gradio |
| API | FastAPI + Uvicorn + SSE 流式 |
| 配置 | YAML/JSON + OmegaConf + HfArgumentParser |
| 构建 | Hatchling（`pyproject.toml`） |

## 目录结构

```
LLaMA-Factory/
├── src/llamafactory/     # 核心 Python 包
├── examples/             # 各场景 YAML 配置
├── data/                 # 演示数据 + dataset_info.json 注册表
├── requirements/         # 可选依赖分组
├── scripts/              # vLLM 推理、LoRA 合并等工具
├── docker/               # Docker Compose 配置
├── docs/                 # 官方 Sphinx 文档
├── tests/                # v0 测试
└── tests_v1/             # v1 架构测试
```

## 双架构说明

项目存在两套并行架构，由环境变量 `USE_V1` 控制：

| 架构 | 开关 | 说明 |
|------|------|------|
| **v0（默认）** | 无 | 当前生产使用，`api/webui → chat/eval/train → data/model → hparams` |
| **v1（实验）** | `USE_V1=1` | 插件化训练器、FSDP2、新配置系统，位于 `src/llamafactory/v1/` |

日常使用和文档分析以 **v0 架构** 为主。

## 关键文件速查

| 用途 | 路径 |
|------|------|
| 包元数据 | `pyproject.toml` |
| CLI 入口 | `src/llamafactory/cli.py` |
| 命令路由 | `src/llamafactory/launcher.py` |
| 训练编排 | `src/llamafactory/train/tuner.py` |
| 参数解析 | `src/llamafactory/hparams/parser.py` |
| 模型加载 | `src/llamafactory/model/loader.py` |
| 数据加载 | `src/llamafactory/data/loader.py` |
| 提示模板 | `src/llamafactory/data/template.py` |
| API 服务 | `src/llamafactory/api/app.py` |
| Web UI | `src/llamafactory/webui/interface.py` |
| 数据集注册 | `data/dataset_info.json` |
| 示例配置 | `examples/train_lora/` |

## 依赖版本（核心）

来自 `pyproject.toml`：

| 包 | 版本约束 |
|----|---------|
| torch | ≥ 2.4.0 |
| transformers | ≥ 4.55.0, ≤ 5.6.0 |
| datasets | ≥ 2.16.0, ≤ 4.0.0 |
| accelerate | ≥ 1.3.0, ≤ 1.11.0 |
| peft | ≥ 0.18.0, ≤ 0.18.1 |
| trl | ≥ 0.18.0, ≤ 0.24.0 |
| gradio | ≥ 4.38.0, ≤ 5.50.0 |

可选依赖见 `requirements/` 目录（deepspeed、bitsandbytes、vllm、sglang 等）。
