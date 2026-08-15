# LLaMA-Factory 0.9.6.dev0 源码分析

本目录是对本地源码 `/home/cp/work2/largeModels/02-训练/LLaMA-Factory` 的中文结构化分析。版本依据为 `src/llamafactory/extras/env.py::VERSION = "0.9.6.dev0"`；内容描述的是该快照，而不是 PyPI 稳定版或上游仓库最新分支。

## 范围与约定

- **v0**：默认生产路径，`api/webui/chat/eval/train → data/model → hparams → extras`。
- **v1**：设置 `USE_V1=1` 后进入的实验架构，拥有独立 config/core/plugins/trainers。
- 文中的路径默认相对项目根目录。
- 参数行为以 dataclass、parser 校验和实际调用点为准；WebUI 与示例 YAML 只是配置入口。
- 上游依赖（Transformers、TRL、PEFT、Accelerate）版本变化会改变兼容性，尤其是 PPO、FSDP2、FP8 与 v1。

## 文档索引 00–17

### 入门、架构与配置

| 编号 | 文档 | 内容 |
|---|---|---|
| 00 | [环境搭建指南](./00-环境搭建指南.md) | Python 环境、安装、验证与排错 |
| 01 | [项目概览](./01-LLaMA-Factory项目概览.md) | 项目定位、能力、技术栈 |
| 02 | [架构设计](./02-架构设计.md) | 模块分层、依赖关系、v0/v1 概览 |
| 03 | [CLI 与入口](./03-CLI与入口.md) | console script、launcher、子命令、torchrun |
| 04 | [配置系统](./04-配置系统.md) | YAML/JSON、hparams dataclass、参数校验 |

### 核心流水线

| 编号 | 文档 | 内容 |
|---|---|---|
| 05 | [训练流水线](./05-训练流水线.md) | `run_exp()`、stage 路由、workflow/Trainer |
| 06 | [数据模块](./06-数据模块.md) | dataset info、converter、processor、collator、template |
| 07 | [模型模块](./07-模型模块.md) | tokenizer/model loader、adapter、量化与 patch |
| 08 | [推理与 Chat](./08-推理与Chat.md) | ChatModel、HF/vLLM/SGLang 引擎、API |
| 09 | [Web UI 与 LLaMA Board](./09-Web-UI与LLaMA-Board.md) | Gradio 组件、Runner 子进程、状态监控 |

### 实践、参考与专题

| 编号 | 文档 | 内容 |
|---|---|---|
| 10 | [使用指南](./10-使用指南.md) | 训练、导出、部署示例 |
| 11 | [API 参考](./11-API参考.md) | 关键函数、类和参数速查 |
| 12 | [对齐与偏好训练](./12-对齐与偏好训练.md) | RM/PPO/DPO/KTO、模型角色、`pref_loss` 六变体 |
| 13 | [分布式与硬件加速](./13-分布式与硬件加速.md) | torchrun、DeepSpeed、FSDP、Ray、MCA、HyperParallel、KT、NPU |
| 14 | [v1 实验架构](./14-v1实验架构.md) | v1 CLI、core、plugin、trainer 与 v0 差异 |
| 15 | [评测与工具脚本](./15-评测与工具脚本.md) | legacy eval、WebUI evaluation、NLG/统计/转换脚本 |
| 16 | [高级算法与 Extras](./16-高级算法与Extras.md) | GaLore、APOLLO、BAdam、Muon、PiSSA、LoRA+、FP8、Liger、Unsloth |
| 17 | [源码测验 100 题](./17-源码测验100题.md) | 概念/配置/调用链自测与参考答案 |

## 推荐阅读路径

### 1. 第一次使用

```text
00 环境 → 01 概览 → 10 使用指南 → 04 配置
```

目标是完成安装、运行 LoRA SFT、导出和推理，不必先读完整源码。

### 2. 理解一次训练怎样发生

```text
02 架构 → 03 CLI → 04 配置 → 05 训练
                         ├→ 06 数据
                         └→ 07 模型
```

建议沿 `llamafactory-cli train YAML → launcher → run_exp → workflow → Trainer` 跟踪。

### 3. 对齐训练/RLHF

```text
05 训练 → 06 数据 → 12 对齐训练 → 13 分布式
```

重点区分 RM、PPO、DPO、KTO，以及 reference/reward/value-head 的不同角色。

### 4. 性能与大规模训练

```text
04 配置 → 05 训练 → 13 分布式与硬件 → 16 高级算法
```

先确定进程/分片后端，再选择 optimizer、kernel 或精度优化；不要把 torchrun、FSDP、Liger、FP8 当作同一层。

### 5. 推理、服务与可视化

```text
07 模型 → 08 推理/API → 09 WebUI → 15 评测
```

适合追踪 ChatModel、后端选择、LLaMA Board 子进程和评测产物。

### 6. v1 研究与二次开发

```text
02 双架构概览 → 14 v1 专题 → examples/v1 → tests_v1
```

不要先从 v0 YAML 直接迁移；v1 是独立实现。

### 7. 源码自测

```text
通读 01–16 后 → 17 源码测验 100 题（先做题，再对答案）
```

### 8. 按问题查源码

```text
命令没启动       → 03
参数被拒绝       → 04
数据格式错误     → 06
模型/adapter 问题→ 07
偏好损失问题     → 12
多卡/多节点问题  → 13
v1 行为差异      → 14
指标/脚本选择    → 15
高级算法组合冲突 → 16
```

## 源码地图

```text
LLaMA-Factory/
├── pyproject.toml                    # 包、依赖组、console scripts
├── src/llamafactory/
│   ├── cli.py                        # USE_V1 总开关
│   ├── launcher.py                   # v0 子命令与 torchrun
│   ├── hparams/                      # v0 参数与 parser
│   ├── train/
│   │   ├── tuner.py                  # run_exp、Ray、stage 路由、export
│   │   ├── pt|sft|rm|ppo|dpo|kto/   # 标准 stage
│   │   ├── mca/                      # Megatron Core Adapter
│   │   ├── hyper_parallel/           # HyperParallel 后端
│   │   ├── trainer_utils.py          # 模型角色、optimizer、loss、Ray 工具
│   │   └── fp8_utils.py              # Accelerate FP8
│   ├── data/                         # loader/converter/processor/collator/template
│   ├── model/                        # loader/patcher/adapter/model_utils
│   ├── chat/                         # HF/vLLM/SGLang/KT
│   ├── api/                          # OpenAI-compatible FastAPI
│   ├── webui/                        # LLaMA Board
│   ├── eval/                         # legacy 选择题 Evaluator
│   ├── extras/                       # 常量、设备、日志、版本、依赖检查
│   └── v1/
│       ├── launcher.py
│       ├── config/ accelerator/ core/
│       ├── plugins/
│       ├── trainers/ samplers/
│       └── utils/
├── examples/
│   ├── train_* merge_lora inference/
│   ├── accelerate/ ascend/ megatron/ ktransformers/
│   ├── extras/
│   └── v1/
├── scripts/                          # 评测、统计、转换、初始化工具
├── requirements/                     # 可选功能依赖
├── tests/                            # v0 测试
└── tests_v1/                         # v1 测试
```

## 重要的当前版本修正

以下结论容易被旧教程或目录名称误导：

1. **版本是 `0.9.6.dev0`**，不是稳定的 0.9.6 release。
2. **v0 仍是默认架构**；只有 `USE_V1=1` 才进入 v1。
3. **`llamafactory-cli eval` 当前直接抛 `NotImplementedError`**。`src/llamafactory/eval/` 保留不代表 CLI 可用。
4. **WebUI Evaluation 不调用 legacy Evaluator**；它生成 `stage=sft` 配置，再执行 `llamafactory-cli train` 的 `do_eval/do_predict`。
5. **PPO 固定依赖旧 TRL API**，源码要求 `trl>=0.8.6,<=0.9.6`；且不支持 eval dataset 和断点续训。
6. **v0 DPO 有六种 `pref_loss`**：`sigmoid/hinge/ipo/kto_pair/orpo/simpo`；v1 只有 `sigmoid/orpo/simpo`。
7. **`pref_loss=kto_pair` 不等于 `stage=kto`**：前者是配对偏好损失，后者使用独立好/坏反馈。
8. **DPO 数据层复用 `stage="rm"` 的 pairwise processor**，这不表示 DPO workflow 训练奖励模型。
9. **Ray 是资源调度层**，DeepSpeed/FSDP 是状态分片层；二者可组合但职责不同。
10. **MCA 与 HyperParallel 是替代训练后端**：MCA 支持 PT/SFT/DPO；HyperParallel 路由只支持 PT/SFT。
11. **KT 不是普通数据并行**，当前 adapter 路径只支持单一 LoRA；自动 torchrun 会排除 `USE_KT=1`。
12. **FP8 不是 QLoRA 量化**；当前 parser 禁止 FP8 与模型 4/8-bit quantization 同时开启。
13. **v1 尚未覆盖 v0 全功能**：没有 PPO/KTO、WebUI/API/export，`version/env` 也未实现。
14. **v1 Trainer 不继承 HF Trainer**；其 batching、optimizer、checkpoint 和分布式由 `BaseTrainer`/plugins 自己协调。

## 快速命令

```bash
cd /home/cp/work2/largeModels/02-训练/LLaMA-Factory
pip install -e .

# 查看当前版本（确保没有 USE_V1=1）
llamafactory-cli version

# v0 LoRA SFT
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# v0 DPO
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml

# WebUI / OpenAI-compatible API
llamafactory-cli webui
llamafactory-cli api

# v1 实验 SFT
USE_V1=1 llamafactory-cli sft examples/v1/train_lora/train_lora_sft.yaml
```

## 上游资料

- 仓库：https://github.com/hiyouga/LLaMA-Factory
- 官方文档：https://llamafactory.readthedocs.io
- 许可证：Apache-2.0
