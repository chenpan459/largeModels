# 11 — API 参考

> 源码基线：LLaMA-Factory `0.9.6.dev0`。以下签名按源码原样记录；引号表示仅在 `TYPE_CHECKING` 下导入的前向类型。使用流程见 [10-使用指南](./10-使用指南.md)，WebUI 的状态与路径细节见 [09-Web-UI与LLaMA-Board](./09-Web-UI与LLaMA-Board.md)。

## 1. CLI 表面

入口是 `llamafactory.cli:main`，默认分派到 `llamafactory.launcher:launch`；`USE_V1=1` 时改用 `llamafactory.v1.launcher`。

```text
llamafactory-cli <command> [config.yaml|config.json] [key=value ...]
lmf <command> [config.yaml|config.json] [key=value ...]
```

| 命令 | v0 调用目标 | 状态 |
|---|---|---|
| `train` | `llamafactory.train.tuner.run_exp()` | 可用；必要时 launcher 先启动 torchrun |
| `export` | `llamafactory.train.tuner.export_model()` | 可用 |
| `chat` | `llamafactory.chat.chat_model.run_chat()` | 可用 |
| `api` | `llamafactory.api.app.run_api()` | 可用 |
| `webui` | `llamafactory.webui.interface.run_web_ui()` | 可用 |
| `webchat` | `llamafactory.webui.interface.run_web_demo()` | 可用 |
| `env` | `llamafactory.extras.env.print_env()` | 可用 |
| `version` | launcher 内打印 `VERSION` | 可用 |
| `help` | launcher 内打印 `USAGE` | 可用 |
| `eval` | 无调用目标 | **不可用**，立即抛出 `NotImplementedError("Evaluation will be deprecated in the future.")` |

“eval 将弃用”是当前源码的原文；并非警告后继续执行。生成式评估应通过 `train` 配置的 `do_eval` / `do_predict`，或 Board 的 Evaluate & Predict Tab。

## 2. 配置解析 API

定义模块与导入：

```python
from llamafactory.hparams.parser import (
    get_eval_args,
    get_infer_args,
    get_ray_args,
    get_train_args,
    read_args,
)
```

### 2.1 原始参数

```python
def read_args(
    args: dict[str, Any] | list[str] | None = None,
) -> dict[str, Any] | list[str]
```

- 显式传入 `args` 时原样返回；
- `sys.argv[1]` 以 `.yaml`/`.yml` 结尾时用 OmegaConf 加载并合并后续 `key=value`；
- `.json` 同理；
- 否则返回 `sys.argv[1:]`。

### 2.2 类型化参数

```python
def get_train_args(
    args: dict[str, Any] | list[str] | None = None,
) -> _TRAIN_CLS

def get_infer_args(
    args: dict[str, Any] | list[str] | None = None,
) -> _INFER_CLS

def get_eval_args(
    args: dict[str, Any] | list[str] | None = None,
) -> _EVAL_CLS

def get_ray_args(
    args: dict[str, Any] | list[str] | None = None,
) -> RayArguments
```

这是源码中的精确返回注解；三个 tuple alias 依次表示：

- `_TRAIN_CLS`：`ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments`
- `_INFER_CLS`：`ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments`
- `_EVAL_CLS`：`ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments`

`get_eval_args()` 仍是内部 evaluator 的参数 API，但 CLI `eval` 已被 launcher 阻断。默认严格拒绝未知参数；`ALLOW_EXTRA_ARGS=1` 才允许 v0 parser 忽略额外键。

## 3. 训练与导出 API

### 3.1 顶层入口

定义模块：`llamafactory.train.tuner`

```python
from llamafactory.train.tuner import export_model, run_exp
```

源码签名：

```python
def run_exp(
    args: Optional[dict[str, Any]] = None,
    callbacks: Optional[list["TrainerCallback"]] = None,
) -> None

def export_model(
    args: Optional[dict[str, Any]] = None,
) -> None
```

`run_exp()` 先 `read_args()`，解析 Ray 参数，再调用 `_training_function()`。stage 路由为 `pt`、`sft`、`rm`、`ppo`、`dpo`、`kto`。`args=None` 时从 `sys.argv` 读取；虽然注解只写 dict，内部 `read_args()` 后也能处理 CLI list。

最小 Python 调用必须提供完整训练必需项，例如：

```python
from llamafactory.train.tuner import run_exp

run_exp({
    "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "dataset": "identity",
    "template": "qwen3_nothink",
    "output_dir": "saves/api-example",
    "per_device_train_batch_size": 1,
})
```

`export_model()` 要求 `export_dir`。adapter 与 `export_quantization_bit` 不能同时给出；量化模型不能再合并 adapter。

### 3.2 SFT 工作流

定义模块：`llamafactory.train.sft.workflow`

```python
def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
)
```

它是已经完成参数解析后的内部工作流：加载 tokenizer/processor、模板、数据、模型，构造 `CustomSeq2SeqTrainer`，再按 `do_train` / `do_eval` / `do_predict` 执行。一般集成方应调用 `run_exp()`，而不是手工构造五组 dataclass。

## 4. 数据 API

### 4.1 `get_dataset`

定义模块：`llamafactory.data.loader`

```python
from llamafactory.data.loader import get_dataset
```

```python
def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule"
```

注意精确的 `stage` Literal **没有 `dpo`**。DPO 数据在工作流中按 pairwise/RM 语义传入。返回类型定义在 `llamafactory.data.data_utils`：

```python
class DatasetModule(TypedDict):
    train_dataset: Optional[Union[Dataset, IterableDataset]]
    eval_dataset: Optional[
        Union[Dataset, IterableDataset, dict[str, Dataset]]
    ]
```

不存在旧文档所写的 `predict_dataset` 键；预测复用 `eval_dataset`。

### 4.2 模板

定义模块：`llamafactory.data.template`

```python
from llamafactory.data.template import get_template_and_fix_tokenizer

def get_template_and_fix_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> "Template"
```

未设置 `data_args.template` 时，优先解析 tokenizer 自带 chat template，否则使用 `TEMPLATES["empty"]`；显式模板名不存在会抛 `ValueError`。函数会原地修复 tokenizer 的特殊 token。

### 4.3 数据角色

定义模块：`llamafactory.data.data_utils`

```python
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"
```

它与 HTTP 协议的 `llamafactory.api.protocol.Role` 不同：API 使用 `tool`，数据内部使用 `observation`。

## 5. 模型 API

定义模块：`llamafactory.model.loader`

```python
from llamafactory.model.loader import (
    TokenizerModule,
    load_config,
    load_model,
    load_tokenizer,
)
```

```python
class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]

def load_tokenizer(
    model_args: "ModelArguments",
) -> "TokenizerModule"

def load_config(
    model_args: "ModelArguments",
) -> "PretrainedConfig"

def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel"
```

`load_tokenizer()` 会原地更新 `model_args.model_name_or_path`（例如切换下载 Hub），并尽力加载 processor。`load_model()` 依 config 选择 image-text、seq2seq、text-to-waveform 或 causal LM AutoModel，应用补丁与 adapter；`is_trainable=False` 时关闭梯度并进入 eval 模式。

Adapter 定义模块：`llamafactory.model.adapter`

```python
from llamafactory.model.adapter import init_adapter

def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel"
```

当前实现支持 full、freeze、LoRA、OFT。量化训练只允许 LoRA/OFT；PiSSA 初始化不能直接用于量化模型。

## 6. 推理 API

### 6.1 `Response`

定义模块：`llamafactory.chat.base_engine`

```python
from llamafactory.chat.base_engine import Response

@dataclass
class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]
```

### 6.2 `ChatModel`

定义模块：`llamafactory.chat.chat_model`

```python
from llamafactory.chat.chat_model import ChatModel, run_chat
```

构造函数：

```python
def __init__(
    self,
    args: Optional[dict[str, Any]] = None,
) -> None
```

`infer_backend` 按 `EngineName` 选择 Hugging Face、vLLM 或 SGLang。对象创建独立 asyncio event loop 与 daemon thread，以同步包装异步引擎。

同步方法：

```python
def chat(
    self,
    messages: list[dict[str, str]],
    system: Optional[str] = None,
    tools: Optional[str] = None,
    images: Optional[list["ImageInput"]] = None,
    videos: Optional[list["VideoInput"]] = None,
    audios: Optional[list["AudioInput"]] = None,
    **input_kwargs,
) -> list["Response"]

def stream_chat(
    self,
    messages: list[dict[str, str]],
    system: Optional[str] = None,
    tools: Optional[str] = None,
    images: Optional[list["ImageInput"]] = None,
    videos: Optional[list["VideoInput"]] = None,
    audios: Optional[list["AudioInput"]] = None,
    **input_kwargs,
) -> Generator[str, None, None]

def get_scores(
    self,
    batch_input: list[str],
    **input_kwargs,
) -> list[float]
```

异步方法：

```python
async def achat(
    self,
    messages: list[dict[str, str]],
    system: Optional[str] = None,
    tools: Optional[str] = None,
    images: Optional[list["ImageInput"]] = None,
    videos: Optional[list["VideoInput"]] = None,
    audios: Optional[list["AudioInput"]] = None,
    **input_kwargs,
) -> list["Response"]

async def astream_chat(
    self,
    messages: list[dict[str, str]],
    system: Optional[str] = None,
    tools: Optional[str] = None,
    images: Optional[list["ImageInput"]] = None,
    videos: Optional[list["VideoInput"]] = None,
    audios: Optional[list["AudioInput"]] = None,
    **input_kwargs,
) -> AsyncGenerator[str, None]

async def aget_scores(
    self,
    batch_input: list[str],
    **input_kwargs,
) -> list[float]
```

`tools` 是 JSON 字符串，不是 Python list。`run_chat() -> None` 是文本终端循环，不传多模态参数。

示例：

```python
from llamafactory.chat.chat_model import ChatModel

model = ChatModel({
    "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
    "template": "qwen3_nothink",
    "infer_backend": "huggingface",
})
responses = model.chat(
    [{"role": "user", "content": "你好"}],
    max_new_tokens=128,
    temperature=0.7,
)
print(responses[0].response_text)
```

## 7. FastAPI 应用 API

定义模块：`llamafactory.api.app`

```python
from llamafactory.api.app import create_app, run_api

def create_app(chat_model: "ChatModel") -> "FastAPI"
def run_api() -> None
```

`run_api()` 自行创建 `ChatModel`，再以 Uvicorn 启动；`create_app()` 便于嵌入已有 ASGI 进程。应用启用允许所有 origin/method/header 且 `allow_credentials=True` 的 CORS；公网部署应在反向代理或定制 app 中收紧。

### 7.1 端点

| 方法 | 路径 | 请求/响应 | 可用条件 |
|---|---|---|---|
| GET | `/v1/models` | `ModelList` | 总是 |
| POST | `/v1/chat/completions` | `ChatCompletionRequest` → JSON 或 SSE | `engine.can_generate=True`，否则 405 |
| POST | `/v1/score/evaluation` | `ScoreEvaluationRequest` → `ScoreEvaluationResponse` | `engine.can_generate=False`，否则 405 |

`API_KEY` 非空时，三个端点都使用 HTTP Bearer 校验。

### 7.2 请求协议

以下类均定义于 `llamafactory.api.protocol`：

```python
class URL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"

class MultimodalInputItem(BaseModel):
    type: Literal["text", "image_url", "video_url", "audio_url"]
    text: str | None = None
    image_url: URL | None = None
    video_url: URL | None = None
    audio_url: URL | None = None

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]

class FunctionAvailable(BaseModel):
    type: Literal["function", "code_interpreter"] = "function"
    function: FunctionDefinition | None = None

class Function(BaseModel):
    name: str
    arguments: str

class FunctionCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Function

class ChatMessage(BaseModel):
    role: Role
    content: str | list[MultimodalInputItem] | None = None
    tool_calls: list[FunctionCall] | None = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[FunctionAvailable] | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    presence_penalty: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False

class ScoreEvaluationRequest(BaseModel):
    model: str
    messages: list[str]
    max_length: int | None = None
```

消息规则由 `api.chat._process_request()` 强制：

- 可选 system 只能在首条；
- 去掉 system 后消息数必须为奇数；
- 偶数下标只允许 `user`/`tool`；
- 奇数下标只允许 `assistant`/`function`；
- SSE 不支持 tools，也不支持 `n > 1`；
- `presence_penalty` 被映射到引擎的 `repetition_penalty`。

### 7.3 响应协议

```python
class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: Literal["owner"] = "owner"

class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard] = []

class ChatCompletionMessage(BaseModel):
    role: Role | None = None
    content: str | None = None
    tool_calls: list[FunctionCall] | None = None

class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamResponseChoice]

class ScoreEvaluationResponse(BaseModel):
    id: str
    object: Literal["score.evaluation"] = "score.evaluation"
    model: str
    scores: list[float]
```

非流式 choice 是 `{index, message, finish_reason}`；流式 choice 是 `{index, delta, finish_reason=None}`，最后发送 `[DONE]`。

### 7.4 多模态输入安全

定义模块：`llamafactory.api.common`

```python
SAFE_MEDIA_PATH = os.environ.get(
    "SAFE_MEDIA_PATH",
    os.path.join(os.path.dirname(__file__), "safe_media"),
)
ALLOW_LOCAL_FILES = is_env_enabled("ALLOW_LOCAL_FILES", "1")

def check_lfi_path(path: str) -> None
def check_ssrf_url(url: str) -> None
```

支持 base64 data URL、本地文件和远程 HTTP(S)。本地真实路径必须以 `SAFE_MEDIA_PATH` 的真实路径为前缀；远程主机解析出的地址必须是 global IP。两个常量在模块 import 时求值，进程启动后再修改环境变量不会更新它们。

## 8. 关键枚举与 Literal

### 8.1 `llamafactory.extras.constants`

```python
class AttentionFunction(StrEnum):
    AUTO = "auto"
    DISABLED = "disabled"
    SDPA = "sdpa"
    FA2 = "fa2"
    FA3 = "fa3"

class EngineName(StrEnum):
    HF = "huggingface"
    VLLM = "vllm"
    SGLANG = "sglang"

class DownloadSource(StrEnum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"
    OPENMIND = "om"

class QuantizationMethod(StrEnum):
    BNB = "bnb"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"
    EETQ = "eetq"
    HQQ = "hqq"
    MXFP4 = "mxfp4"
    FP8 = "fp8"

class RopeScaling(StrEnum):
    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    LLAMA3 = "llama3"
```

相关常量：

```python
METHODS = ["full", "freeze", "lora", "oft"]
PEFT_METHODS = {"lora", "oft"}
TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "KTO": "kto",
    "Pre-Training": "pt",
}
```

`FinetuningArguments` 的精确 Literal：

```python
stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = "sft"
finetuning_type: Literal["lora", "oft", "freeze", "full"] = "lora"
pref_loss: Literal[
    "sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"
] = "sigmoid"
```

### 8.2 `llamafactory.api.protocol`

```python
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"

class Finish(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL = "tool_calls"
```

## 9. 环境变量

以下均来自当前 v0 源码；布尔变量由 `is_env_enabled()` 解析，真值为不区分大小写的 `true`、`y`、`1`。

### 9.1 入口、解析与运行时

| 变量 | 默认值 | 精确作用 |
|---|---:|---|
| `USE_V1` | `0` | `cli.main()` 选择 v1 launcher |
| `USE_MCA` | `0` | 使用 Megatron-core adapter 参数/工作流，并强制 torchrun |
| `USE_RAY` | `0` | launcher 不做本地自动多卡，由 Ray 工作流处理 |
| `USE_KT` | `0` | KTransformers 模式，launcher 不做本地自动多卡 |
| `ALLOW_EXTRA_ARGS` | `0` | v0 HfArgumentParser 允许额外键 |
| `LLAMAFACTORY_VERBOSITY` | `INFO` | 项目与 Transformers 日志级别 |
| `DISABLE_VERSION_CHECK` | `0` | 对非 mandatory 依赖跳过版本检查 |
| `FORCE_CHECK_IMPORTS` | `0` | 强制执行可选 import 检查 |
| `NPU_JIT_COMPILE` | `0` | NPU `set_compile_mode(jit_compile=...)` |
| `MAX_CONCURRENT` | `1` | Hugging Face 推理引擎 semaphore 容量 |
| `RECORD_VRAM` | `0` | 训练 callback 记录显存 |

### 9.2 分布式 launcher

| 变量 | 默认值 | 精确作用 |
|---|---:|---|
| `FORCE_TORCHRUN` | `0` | 即使单设备也进入 torchrun 分支 |
| `NNODES` | `1` | 节点数 |
| `NODE_RANK` | `0` | 当前节点 rank |
| `NPROC_PER_NODE` | 检测到的设备数 | 每节点进程数 |
| `MASTER_ADDR` | `127.0.0.1` | rendezvous/master 地址 |
| `MASTER_PORT` | 自动空闲端口 | master 端口 |
| `RDZV_ID` | 未设置 | 设置后使用 elastic c10d rendezvous |
| `MIN_NNODES` / `MAX_NNODES` | 未设置 | 两者同时设置时形成 elastic 节点范围 |
| `MAX_RESTARTS` | `0` | elastic 最大重启次数 |
| `OPTIM_TORCH` | `1` | 子进程设置 expandable allocator 与 NCCL record-stream 优化 |

设备可见性（如 `CUDA_VISIBLE_DEVICES`、`ASCEND_RT_VISIBLE_DEVICES`）由 PyTorch/设备运行时消费，LLaMA-Factory 通过检测结果决定进程数。

### 9.3 模型 Hub 与占位符

| 变量 | 默认值 | 精确作用 |
|---|---:|---|
| `USE_MODELSCOPE_HUB` | `0` | 从 ModelScope 选择模型/数据来源 |
| `USE_OPENMIND_HUB` | `0` | 从 OpenMind 选择模型/数据来源 |
| `IMAGE_PLACEHOLDER` | `<image>` | 多模态文本图片占位符 |
| `VIDEO_PLACEHOLDER` | `<video>` | 视频占位符 |
| `AUDIO_PLACEHOLDER` | `<audio>` | 音频占位符 |
| `DOWNSAMPLE_MODE` | 未设置 | 部分多模态插件的下采样模式 |

`HF_ENDPOINT`、`HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE` 等属于 Hugging Face 依赖的环境接口，不由 v0 LLaMA-Factory 自行解析。

### 9.4 HTTP API

| 变量 | 默认值 | 精确作用 |
|---|---:|---|
| `API_HOST` | `0.0.0.0` | Uvicorn host |
| `API_PORT` | `8000` | Uvicorn port，按 int 解析 |
| `API_KEY` | 未设置 | 非空时启用 Bearer 校验 |
| `API_MODEL_NAME` | `gpt-3.5-turbo` | `/v1/models` 唯一 ModelCard ID |
| `API_VERBOSE` | `1` | 记录完整请求 JSON；可能含敏感内容 |
| `FASTAPI_ROOT_PATH` | 空字符串 | FastAPI `root_path` |
| `SAFE_MEDIA_PATH` | `api/safe_media` | 允许读取本地媒体的目录 |
| `ALLOW_LOCAL_FILES` | `1` | 是否允许本地媒体路径 |

### 9.5 WebUI 与 Board

| 变量 | 默认值 | 精确作用 |
|---|---:|---|
| `GRADIO_IPV6` | `0` | 默认监听地址改为 `[::]` |
| `GRADIO_SHARE` | `0` | Gradio `share=True` |
| `GRADIO_SERVER_NAME` | `0.0.0.0` 或 `[::]` | 显式监听地址 |
| `GRADIO_SERVER_PORT` | Gradio 默认 | 由 Gradio 消费，Board 源码未读取 |
| `DEMO_MODEL` | 未设置 | demo mode 自动加载的模型 |
| `DEMO_TEMPLATE` | 未设置 | demo mode 自动加载的模板 |
| `DEMO_BACKEND` | `huggingface` | demo 推理后端 |
| `LLAMABOARD_ENABLED` | Runner 设置为 `1` | LogCallback 进入 Board 模式 |
| `LLAMABOARD_WORKDIR` | Runner 输出目录 | LogCallback 写日志的位置 |

## 10. WebUI Python API

这些对象主要是内部 Gradio glue API，兼容性弱于训练/推理入口。

### 10.1 Interface

定义模块：`llamafactory.webui.interface`

```python
def create_ui(demo_mode: bool = False) -> "gr.Blocks"
def create_web_demo() -> "gr.Blocks"
def run_web_ui() -> None
def run_web_demo() -> None
```

### 10.2 Engine 与 Manager

定义模块：`llamafactory.webui.engine`

```python
class Engine:
    def __init__(
        self,
        demo_mode: bool = False,
        pure_chat: bool = False,
    ) -> None

    def resume(self)
    def change_lang(self, lang: str)
```

定义模块：`llamafactory.webui.manager`

```python
class Manager:
    def __init__(self) -> None
    def add_elems(
        self,
        tab_name: str,
        elem_dict: dict[str, "Component"],
    ) -> None
    def get_elem_list(self) -> list["Component"]
    def get_elem_iter(
        self,
    ) -> Generator[tuple[str, "Component"], None, None]
    def get_elem_by_id(self, elem_id: str) -> "Component"
    def get_id_by_elem(self, elem: "Component") -> str
    def get_base_elems(self) -> set["Component"]
```

### 10.3 Runner

定义模块：`llamafactory.webui.runner`

源码公开方法保持未注解签名：

```python
class Runner:
    def __init__(
        self,
        manager: "Manager",
        demo_mode: bool = False,
    ) -> None

    def set_abort(self) -> None
    def preview_train(self, data)
    def preview_eval(self, data)
    def run_train(self, data)
    def run_eval(self, data)
    def monitor(self)
    def save_args(self, data)
    def load_args(self, lang: str, config_path: str)
    def check_output_dir(
        self,
        lang: str,
        model_name: str,
        finetuning_type: str,
        output_dir: str,
    )
```

`run_eval()` 并不调用已禁用的 CLI `eval`；Runner 始终生成 YAML，再以
`Popen(["llamafactory-cli", "train", generated_training_args_path])` 启动训练入口。

## 11. 源码定位

| 领域 | 定义文件 |
|---|---|
| CLI | `src/llamafactory/cli.py`, `src/llamafactory/launcher.py` |
| 参数 | `src/llamafactory/hparams/parser.py`, `finetuning_args.py` |
| 训练/导出 | `src/llamafactory/train/tuner.py`, `train/sft/workflow.py` |
| 数据/模板 | `src/llamafactory/data/loader.py`, `src/llamafactory/data/template.py`, `src/llamafactory/data/data_utils.py` |
| 模型/adapter | `src/llamafactory/model/loader.py`, `model/adapter.py` |
| 推理 | `src/llamafactory/chat/chat_model.py`, `chat/base_engine.py` |
| HTTP | `src/llamafactory/api/app.py`, `api/chat.py`, `api/protocol.py`, `api/common.py` |
| WebUI | `src/llamafactory/webui/interface.py`, `engine.py`, `manager.py`, `runner.py` |
| 枚举 | `src/llamafactory/extras/constants.py`, `src/llamafactory/api/protocol.py`, `src/llamafactory/data/data_utils.py` |
