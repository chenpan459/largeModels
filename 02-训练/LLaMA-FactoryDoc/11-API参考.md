# 11 — API 参考

## CLI 命令

```bash
llamafactory-cli <command> [config.yaml] [key=value ...]
lmf <command> [config.yaml] [key=value ...]    # 快捷别名
```

| 命令 | 函数 | 说明 |
|------|------|------|
| `train` | `train.tuner.run_exp()` | 训练 |
| `export` | `train.tuner.export_model()` | 导出/合并 |
| `api` | `api.app.run_api()` | API 服务 |
| `chat` | `chat.chat_model.run_chat()` | CLI 对话 |
| `webui` | `webui.interface.run_web_ui()` | Web UI |
| `webchat` | `webui.interface.run_web_demo()` | Web 对话 |
| `env` | `extras.env.print_env()` | 环境信息 |
| `version` | — | 版本信息 |

## 配置解析

### read_args

```python
# src/llamafactory/hparams/parser.py
def read_args(args: dict | list[str] | None = None) -> dict | list[str]
```

从 YAML/JSON/CLI 读取原始参数。

### get_train_args

```python
def get_train_args(args) -> tuple[
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
]
```

解析并校验训练参数，返回五个 dataclass。

### get_infer_args

```python
def get_infer_args(args) -> tuple[
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
]
```

解析推理/导出参数。

## 训练 API

### run_exp

```python
# src/llamafactory/train/tuner.py
def run_exp(
    args: dict[str, Any] | None = None,
    callbacks: list[TrainerCallback] | None = None,
) -> None
```

训练总入口。`args` 为配置字典，不传则从 `sys.argv` 读取。

```python
from llamafactory.train.tuner import run_exp

run_exp({"model_name_or_path": "Qwen/Qwen3-4B", "stage": "sft", ...})
```

### run_sft

```python
# src/llamafactory/train/sft/workflow.py
def run_sft(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    generating_args: GeneratingArguments,
    callbacks: list[TrainerCallback] | None = None,
) -> None
```

SFT 工作流：加载 tokenizer → 模板 → 数据集 → 模型 → Trainer → 训练 → 保存。

### export_model

```python
# src/llamafactory/train/tuner.py
def export_model(args: dict[str, Any] | None = None) -> None
```

合并 LoRA 并导出完整模型到 `export_dir`。

## 数据 API

### get_dataset

```python
# src/llamafactory/data/loader.py
def get_dataset(
    template: Template,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"],
    **tokenizer_module,
) -> DatasetModule
```

返回 `{"train_dataset", "eval_dataset", "predict_dataset"}`。

### get_template_and_fix_tokenizer

```python
# src/llamafactory/data/template.py
def get_template_and_fix_tokenizer(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
) -> Template
```

选择模板并修复 tokenizer 特殊 token。

## 模型 API

### load_tokenizer

```python
# src/llamafactory/model/loader.py
def load_tokenizer(model_args: ModelArguments) -> TokenizerModule
# TokenizerModule = {"tokenizer": PreTrainedTokenizer, "processor": ProcessorMixin | None}
```

### load_model

```python
def load_model(
    tokenizer: PreTrainedTokenizer,
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> PreTrainedModel
```

加载模型并注入适配器（LoRA/OFT/Full/Freeze）。

### init_adapter

```python
# src/llamafactory/model/adapter.py
def init_adapter(
    config: PretrainedConfig,
    model: PreTrainedModel,
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
    is_trainable: bool,
) -> PreTrainedModel
```

## 推理 API

### ChatModel

```python
# src/llamafactory/chat/chat_model.py
class ChatModel:
    def __init__(self, args: dict | None = None) -> None

    # 同步
    def chat(self, messages, system=None, tools=None, images=None, ...) -> list[Response]
    def stream_chat(self, messages, ...) -> Generator[str, None, None]
    def get_scores(self, batch_input, ...) -> list[float]

    # 异步
    async def achat(self, messages, ...) -> list[Response]
    async def astream_chat(self, messages, ...) -> AsyncGenerator[str, None]
    async def aget_scores(self, batch_input, ...) -> list[float]
```

### run_chat

```python
def run_chat() -> None
```

启动 CLI 交互式对话（从 `sys.argv` 读取推理配置）。

## HTTP API

### FastAPI 端点

| 方法 | 路径 | 请求体 | 响应 |
|------|------|--------|------|
| GET | `/v1/models` | — | `ModelList` |
| POST | `/v1/chat/completions` | `ChatCompletionRequest` | `ChatCompletionResponse` 或 SSE |
| POST | `/v1/score/evaluation` | `ScoreEvaluationRequest` | `ScoreEvaluationResponse` |

### ChatCompletionRequest

```python
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    tools: list[Function] | None = None
```

## 关键枚举

### 训练阶段（stage）

| 值 | 说明 |
|----|------|
| `pt` | 继续预训练 |
| `sft` | 监督微调 |
| `rm` | 奖励模型 |
| `ppo` | PPO 强化学习 |
| `dpo` | 直接偏好优化 |
| `kto` | KTO 优化 |

### 微调类型（finetuning_type）

| 值 | 说明 |
|----|------|
| `lora` | Low-Rank Adaptation |
| `oft` | Orthogonal Fine-Tuning |
| `freeze` | 部分层微调 |
| `full` | 全量微调 |

### 推理后端（infer_backend）

| 值 | 说明 |
|----|------|
| `huggingface` | Transformers 原生 |
| `vllm` | vLLM 加速 |
| `sglang` | SGLang 结构化生成 |

### DPO 损失（pref_loss）

| 值 | 说明 |
|----|------|
| `sigmoid` | 标准 DPO |
| `hinge` | Hinge 损失 |
| `ipo` | Identity Preference Optimization |
| `orpo` | Odds Ratio Preference Optimization |
| `simpo` | Simple Preference Optimization |
| `kto_pair` | KTO 成对变体 |

## 环境变量

| 变量 | 说明 |
|------|------|
| `USE_V1` | 切换到 v1 实验架构 |
| `USE_MCA` | 启用 Megatron-core |
| `FORCE_TORCHRUN` | 强制 torchrun |
| `OPTIM_TORCH` | 优化 DDP 内存 |
| `API_HOST` / `API_PORT` / `API_KEY` | API 服务配置 |
| `GRADIO_SERVER_PORT` | Web UI 端口 |
| `HF_ENDPOINT` | HuggingFace 镜像 |
| `WANDB_DISABLED` | 禁用 W&B |
| `CUDA_VISIBLE_DEVICES` | 指定 GPU |

## Web UI API

### Engine

```python
# src/llamafactory/webui/engine.py
class Engine:
    manager: Manager     # UI 元素管理
    runner: Runner       # 训练子进程
    chatter: WebChatModel  # 对话模型

    def resume(self) -> dict          # 恢复上次配置
    def change_lang(self, lang) -> dict  # 切换语言
```

### Runner

```python
# src/llamafactory/webui/runner.py
class Runner:
    def run_train(self, data) -> Generator   # 启动训练
    def run_eval(self, data) -> Generator    # 启动评测
    def set_abort(self) -> None              # 中止运行
    def monitor(self) -> Generator           # 监控日志
```

## 文件路径索引

| 模块 | 关键文件 |
|------|---------|
| CLI | `src/llamafactory/cli.py`, `launcher.py` |
| 训练 | `src/llamafactory/train/tuner.py`, `sft/workflow.py` |
| 参数 | `src/llamafactory/hparams/parser.py`, `finetuning_args.py` |
| 数据 | `src/llamafactory/data/loader.py`, `template.py` |
| 模型 | `src/llamafactory/model/loader.py`, `adapter.py` |
| 推理 | `src/llamafactory/chat/chat_model.py`, `hf_engine.py` |
| API | `src/llamafactory/api/app.py`, `chat.py`, `protocol.py` |
| Web UI | `src/llamafactory/webui/interface.py`, `engine.py`, `runner.py` |
| 常量 | `src/llamafactory/extras/constants.py` |
