# 09 — Web UI 与 LLaMA Board

> 源码基线：LLaMA-Factory `0.9.6.dev0`。本章解释 Board 自身的控制流与磁盘布局；训练参数、数据格式和公开 Python/HTTP API 分别见 [10-使用指南](./10-使用指南.md)、[06-数据模块](./06-数据模块.md) 和 [11-API参考](./11-API参考.md)。

## 1. 两种界面与启动方式

在仓库根目录执行：

```bash
# 完整 Board：Train、Evaluate & Predict、Chat、Export
llamafactory-cli webui

# 纯对话界面；模型参数从命令行配置读取
llamafactory-cli webchat examples/inference/qwen3_lora_sft.yaml

# 仓库中确实存在的等价完整 Board 脚本
python src/webui.py
```

`webui` 由 `run_web_ui()` 调用 `create_ui().queue().launch(...)`；`webchat` 由 `run_web_demo()` 调用 `create_web_demo()`。默认 Gradio 端口是 7860（Gradio 自身默认值），源码只显式设置监听地址。

常用环境变量：

| 变量 | 行为 |
|---|---|
| `GRADIO_SERVER_NAME` | 监听地址，默认 `0.0.0.0`；启用 IPv6 时默认 `[::]` |
| `GRADIO_IPV6=1` | 使用 IPv6，并由 `fix_proxy()` 调整本地代理绕过 |
| `GRADIO_SHARE=1` | 把 `share=True` 传给 Gradio |
| `GRADIO_SERVER_PORT` | Gradio 识别的端口变量；不是 `interface.py` 自行解析 |
| `DEMO_MODEL`、`DEMO_TEMPLATE` | `demo_mode=True` 时自动加载演示模型 |
| `DEMO_BACKEND` | 演示推理后端，默认 `huggingface` |

## 2. 模块分层

```text
launcher.py
  └─ webui/interface.py          组装 Blocks、Tab 和事件
       └─ Engine
          ├─ Manager             Component ↔ "tab.name" 双向索引
          ├─ Runner              训练/预测子进程、状态与监控
          └─ WebChatModel        进程内推理、流式对话

components/
  ├─ top.py                      全局模型、适配器、模板、量化选项
  ├─ train.py / eval.py          训练和预测事件
  ├─ infer.py / chatbot.py       模型装卸、文本/工具/多模态对话
  ├─ export.py                   进程内调用 export_model()
  ├─ data.py                     数据集预览
  └─ footer.py                   设备显存显示
```

训练和 Evaluate & Predict 共用 `Runner`，但 Chat 与 Export **不启动训练子进程**：Chat 在 Web 进程中构造 `ChatModel`，Export 在 Web 进程中直接调用 `train.tuner.export_model()`。

## 3. Interface：界面装配

`src/llamafactory/webui/interface.py` 的 `create_ui(demo_mode=False)` 按以下顺序注册组件：

1. `head`：标题和副标题；
2. `top`：四个业务 Tab 共用的基础参数；
3. `train`、`eval`、`infer`；
4. 非 demo 模式下的 `export`；
5. `footer`。

页面 load 事件连接 `engine.resume`，语言 change 事件连接 `engine.change_lang`，语言 input 事件连接 `save_config`。完整 Board 的 `demo_mode=True` 仍显示训练、预测和聊天，但 Runner 会拒绝真正启动任务，且隐藏 Export；`create_web_demo()` 则是仅含聊天框的 `pure_chat=True` 界面。

## 4. Manager：组件地址空间

`Manager.add_elems(tab_name, elem_dict)` 把组件注册成稳定 ID，例如：

- `top.model_path`
- `train.dataset`
- `eval.output_box`
- `infer.chat_box`

内部同时维护 `_id_to_elem` 与 `_elem_to_id`，因此 Runner 既能通过 ID 从 Gradio 的 `data` 字典取值，也能把组件反查为 ID 后持久化。

`get_base_elems()` 是各业务 Tab 的共享输入集合，包括语言、模型名/路径、微调类型、检查点、量化位数/方法、模板、RoPE scaling 和 booster。组件新增或改名时，必须同步检查：

1. `components/*.py` 的注册键；
2. `Manager.get_base_elems()`；
3. `Runner._parse_train_args()` / `_parse_eval_args()`；
4. `Engine.resume()` 的 ID；
5. `locales.py` 中以末段组件名为键的翻译。

`get_elem_iter()` 故意只向国际化层暴露 ID 最后一段，例如 `train.dataset` 和 `eval.dataset` 都映射到 locale 键 `dataset`。

## 5. Engine：生命周期协调

`Engine.__init__(demo_mode=False, pure_chat=False)` 创建：

```text
Manager()
Runner(manager, demo_mode)
WebChatModel(manager, demo_mode, lazy_init=not pure_chat)
```

非 demo 模式还会调用 `create_ds_config()`，在当前工作目录生成 Board 使用的 DeepSpeed 配置。

### `resume()` 的真实含义

页面首次加载时，`resume()`：

- 从 `llamaboard_cache/user_config.yaml` 恢复语言、模型 Hub 和上次模型；
- 为训练、预测生成当前时间和默认输出名；
- 根据 `self.chatter.loaded` 恢复聊天框可见性；
- 若 Python 进程内的 Runner 仍持有运行中的子进程，则恢复上次组件值，显示隐藏的 `resume_btn`，并重新进入 `monitor()`。

这里的“恢复”是 **刷新页面后重新挂接正在运行的子进程**，不是从磁盘跨 Web 进程恢复任务。训练断点续训由输出目录及训练参数解析完成，见第 7 节。

`change_lang(lang)` 遍历组件，使用 `LOCALES[组件末段名][lang]` 构造同类 Gradio 组件更新；支持 `en`、`ru`、`zh`、`ko`、`ja`。

## 6. Components：Tab 与事件

### 6.1 顶部公共区

`components/top.py` 提供：

- 模型来源：Hugging Face、ModelScope、OpenMind；
- 内置模型或 `Custom` 路径；
- `full`、`freeze`、`lora`、`oft`；
- 保存目录中的检查点列表；
- BNB/HQQ/EETQ 量化位数；
- 模板、RoPE scaling、FlashAttention 2/Unsloth/Liger booster。

切换 Hub 会设置 `USE_MODELSCOPE_HUB` / `USE_OPENMIND_HUB`，刷新模型路径和模板。量化只对 `PEFT_METHODS={"lora", "oft"}` 开放；具体可选位数由 `QuantizationMethod` 决定。

### 6.2 Train

`components/train.py` 覆盖六类输入：

1. stage、数据目录/数据集和数据预览；
2. 学习率、epoch、长度、batch、梯度累积和精度；
3. logging/save/warmup、packing、模板行为和实验跟踪；
4. freeze、LoRA、DPO/KTO/PPO 参数；
5. 多模态冻结策略与图像/视频像素限制；
6. GaLore、APOLLO、BAdam、SwanLab、DeepSpeed。

按钮分别连接 `preview_train`、`save_args`、`load_args`、`run_train` 和 `set_abort`。`extra_args` 必须是 JSON 对象，其键值最后覆盖 Board 生成的同名参数。

### 6.3 Evaluate & Predict

`components/eval.py` 实际是以 `stage=sft` 调用训练入口：

- `predict=true` → `do_predict=true`；
- `predict=false` → `do_eval=true`；
- 始终设置 `eval_dataset`；
- 始终设置 `predict_with_generate=true`。

因此它是数据集上的 SFT 生成式预测/评估，不是旧 `llamafactory-cli eval` benchmark 命令。该 CLI 分支目前直接抛出 `NotImplementedError("Evaluation will be deprecated in the future.")`，不要把两者混为一谈。

### 6.4 Chat

`components/infer.py` 支持 `huggingface`、`vllm`、`sglang`，先由 `WebChatModel.load_model()` 解析公共参数和 `infer.extra_args`，再在当前 Web 进程构造推理引擎。

`components/chatbot.py` 支持：

- `user` / `observation` 输入角色；
- system prompt 与 JSON 工具定义；
- image、video、audio 单项上传；
- 流式输出、思考内容折叠、特殊 token 和 HTML escaping。

多模态框只在所选模型属于 `MULTIMODAL_SUPPORTED_MODELS` 时显示。`WebChatModel.stream()` 将媒体分别包装为 `images`、`videos`、`audios`，临时修改模板的 `enable_thinking`，并把工具调用转成 `function` 消息。

### 6.5 Export

`components/export.py` 收集分片大小、GPTQ 位数/校准数据、设备、legacy format、目标目录和 Hub ID，校验后同步调用 `export_model(args)`。

约束包括：

- 合并 PEFT adapter 时必须选择检查点；
- GPTQ 量化必须提供校准数据；
- 不能在一次操作中同时合并 LoRA 和 GPTQ 量化；
- 导出期间占用 Web 进程，完成后调用 `torch_gc()`。

## 7. Runner：从表单到子进程

### 7.1 参数构造与校验

`_initialize()` 拒绝并发任务、缺失模型/路径/数据集/输出目录、非法 JSON、PPO 缺少奖励模型，以及 demo 模式的真实启动。没有 CUDA/NPU 等 accelerator 时只给警告，不阻止执行。

`_parse_train_args()` 把 UI 值转换为训练参数。几个容易忽略的规则：

- Board 输出目录通常是 `saves/<model_name>/<finetuning_type>/<output_dir>`；
- 若 `output_dir` 本身含路径分隔符，`get_save_dir()` 原样使用该路径；
- PEFT 检查点写入逗号分隔的 `adapter_name_or_path`；full/freeze 检查点替换 `model_name_or_path`；
- `val_size > 1e-6` 且 stage 非 PPO 时，自动设置 `eval_strategy=steps`、`eval_steps=save_steps`；
- 多模态像素值允许 `768*768` 形式，落参前转为整数；
- 选择 DeepSpeed 时，使用 `llamaboard_cache/ds_z{2|3}[_offload]_config.json`。

### 7.2 启动链路

```text
Start
  → _initialize()
  → _parse_train_args() / _parse_eval_args()
  → 创建 output_dir
  → 保存 output_dir/llamaboard_config.yaml
  → save_cmd() 写 output_dir/training_args.yaml
  → Popen(["llamafactory-cli", "train", training_args.yaml])
  → monitor()
```

子进程环境额外包含：

- `LLAMABOARD_ENABLED=1`
- `LLAMABOARD_WORKDIR=<output_dir>`
- 使用 DeepSpeed 时 `FORCE_TORCHRUN=1`

stdout 和 stderr 都追加到 `<output_dir>/webui_subprocess.log`。源码明确使用参数列表且不启用 `shell=True`。

### 7.3 监控、停止与续训

`monitor()` 每 2 秒通过 `Popen.communicate(timeout=2)` 检查进程，并由 `control.get_trainer_info()` 读取：

- `running_log.txt`：界面仅展示末尾 20,000 字符；
- `trainer_log.jsonl`：进度、耗时和 loss 图；
- `swanlab_public_config.json`：实验链接。

停止按钮递归对子进程树发送 `SIGABRT`。失败时优先展示 `communicate()` 得到的 stderr；由于日志被重定向，通常回退读取 `webui_subprocess.log` 最后 20,000 字节。

输出目录输入事件会读取 `<output_dir>/llamaboard_config.yaml` 并还原表单。真正训练时，参数解析器还会在已有 `output_dir` 中自动找到最后一个 `checkpoint-*` 并设置 `resume_from_checkpoint`；但配置中的 `overwrite_output_dir: true` 会绕开自动续训，显式续训方式见 [10-使用指南](./10-使用指南.md)。

## 8. 磁盘与缓存路径

所有相对路径都相对 **启动 Board 时的当前工作目录**，不是用户主目录：

| 路径 | 生产者 | 内容 |
|---|---|---|
| `llamaboard_cache/user_config.yaml` | `save_config()` | 语言、Hub、上次模型、自定义模型路径、可选 cache_dir |
| `llamaboard_cache/ds_z2_config.json` | `create_ds_config()` | ZeRO-2 |
| `llamaboard_cache/ds_z2_offload_config.json` | 同上 | ZeRO-2 optimizer CPU offload |
| `llamaboard_cache/ds_z3_config.json` | 同上 | ZeRO-3 |
| `llamaboard_cache/ds_z3_offload_config.json` | 同上 | ZeRO-3 optimizer/parameter CPU offload |
| `llamaboard_config/<name>.yaml` | Save arguments | Board 组件 ID 到值的映射，不是直接训练参数 |
| `saves/<model>/<method>/<run>/llamaboard_config.yaml` | `_launch()` | 本次运行的 Board 表单快照 |
| 同目录 `training_args.yaml` | `save_cmd()` | 子进程真正读取的清理后训练参数 |
| 同目录 `webui_subprocess.log` | `_launch()` | 子进程合并日志 |
| 同目录 `running_log.txt`、`trainer_log.jsonl` | `LogCallback` | Board 日志、进度和 loss 数据 |

旧文档中“保存在 `~/.cache/llamafactory/`”的说法不适用于此版本源码。

## 9. 文件索引

| 文件 | 核心职责 |
|---|---|
| `webui/interface.py` | 完整/纯聊天 Blocks 和 launch |
| `webui/engine.py` | 生命周期、初始状态、语言更新 |
| `webui/manager.py` | 组件双向索引 |
| `webui/runner.py` | 参数转换、子进程、监控、配置快照 |
| `webui/chatter.py` | Web 模型装卸和流式对话 |
| `webui/control.py` | Hub、列表、量化能力、日志读取 |
| `webui/common.py` | 路径、YAML、命令、进程终止、DS 配置 |
| `webui/components/*.py` | 各界面区块及 Gradio 事件绑定 |
| `webui/locales.py` | 组件文案和告警国际化 |
| `webui/css.py` | 样式 |
