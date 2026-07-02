# 09 — Web UI（LLaMA Board）

## 模块概览

Web UI 位于 `src/llamafactory/webui/`，基于 Gradio 构建可视化操作界面 LLaMA Board。

```
interface.py     ← UI 构建（Tab 布局）
engine.py        ← 中央控制器
runner.py        ← 训练/评测子进程管理
chatter.py       ← Web 对话模型
components/      ← 各 Tab 的 UI 组件
locales.py       ← 国际化（中/英/日/韩/俄）
common.py        ← 配置保存/加载
```

## 启动方式

```bash
# 完整 LLaMA Board（训练 + 评测 + 对话 + 导出）
llamafactory-cli webui

# 仅对话 Web Demo
llamafactory-cli webchat

# 或直接运行脚本
python src/webui.py
```

默认监听 `0.0.0.0:7860`，可通过环境变量 `GRADIO_SERVER_PORT` 修改端口。

## 界面结构

LLaMA Board 包含四个 Tab：

```python
# interface.py
with gr.Tab("Train"):
    create_train_tab(engine)          # 训练
with gr.Tab("Evaluate & Predict"):
    create_eval_tab(engine)           # 评测与预测
with gr.Tab("Chat"):
    create_infer_tab(engine)          # 对话
with gr.Tab("Export"):
    create_export_tab(engine)         # 导出
```

### Train Tab

`components/train.py` — 可视化配置并启动训练：

| 配置区 | 内容 |
|--------|------|
| 模型设置 | 模型路径、适配器路径、量化 |
| 训练方法 | stage、finetuning_type、LoRA 参数 |
| 数据集 | 数据集选择、模板、截断长度 |
| 训练参数 | batch size、学习率、epoch、精度 |
| 输出 | 输出目录、日志步数、保存步数 |

点击「Start」后，`Runner` 将 UI 配置转为 YAML 并以子进程启动训练。

### Evaluate & Predict Tab

`components/eval.py` — MMLU 等 benchmark 评测和批量预测。

### Chat Tab

`components/infer.py` + `components/chatbot.py` — 加载模型并进行对话测试，支持流式输出和多模态输入。

### Export Tab

`components/export.py` — 合并 LoRA 适配器并导出完整模型。

## 核心组件

### Engine（中央控制器）

`webui/engine.py` 的 `Engine` 类协调三个子系统：

```python
class Engine:
    def __init__(self, demo_mode=False, pure_chat=False):
        self.manager = Manager()       # UI 元素管理
        self.runner = Runner(...)      # 训练/评测子进程
        self.chatter = WebChatModel()  # Web 对话
```

| 组件 | 文件 | 职责 |
|------|------|------|
| `Manager` | `manager.py` | 管理 Gradio 元素 ID 映射 |
| `Runner` | `runner.py` | 启动/停止/监控训练子进程 |
| `WebChatModel` | `chatter.py` | Web 端对话模型封装 |

### Runner 工作流程

```
用户点击 Start
    ↓
Runner.save_args()          # UI 值 → YAML 配置
    ↓
Runner._launch()            # subprocess: llamafactory-cli train config.yaml
    ↓
Runner.monitor()            # 读取 stdout，更新 UI 日志
    ↓
训练完成 → 更新 UI 状态
```

Runner 支持：
- 启动/中止训练
- 实时显示训练日志
- 断点续训
- 多 GPU 自动 torchrun

### 国际化

`webui/locales.py` 支持五种语言：

| 语言 | 代码 |
|------|------|
| English | `en` |
| 中文 | `zh` |
| 日本語 | `ja` |
| 한국어 | `ko` |
| Русский | `ru` |

切换语言通过顶部 Dropdown 控件，选择后所有 UI 文本即时更新。

## 配置持久化

`webui/common.py` 负责配置的保存与加载：

- 用户配置保存在 `~/.cache/llamafactory/` 目录
- 页面加载时自动恢复上次配置（`engine.resume()`）
- 语言偏好也会持久化

## Web Demo 模式

`create_web_demo()` 创建精简版界面，仅包含对话功能：

```bash
llamafactory-cli webchat
```

适用于 HuggingFace Spaces 等在线 Demo 部署，不包含训练和导出功能。

## 关键文件

| 文件 | 说明 |
|------|------|
| `webui/interface.py` | UI 构建入口，`create_ui()` / `run_web_ui()` |
| `webui/engine.py` | Engine 中央控制器 |
| `webui/runner.py` | 训练/评测子进程管理 |
| `webui/chatter.py` | Web 对话模型 |
| `webui/manager.py` | Gradio 元素 ID 管理 |
| `webui/common.py` | 配置保存/加载、命令生成 |
| `webui/locales.py` | 多语言翻译 |
| `webui/css.py` | 自定义样式 |
| `webui/components/train.py` | 训练 Tab |
| `webui/components/eval.py` | 评测 Tab |
| `webui/components/infer.py` | 推理 Tab |
| `webui/components/chatbot.py` | 对话组件 |
| `webui/components/export.py` | 导出 Tab |
