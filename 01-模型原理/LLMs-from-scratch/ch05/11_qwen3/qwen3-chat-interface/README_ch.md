# 从零实现 Qwen3 与聊天界面

本 bonus 文件夹包含运行类 ChatGPT 用户界面的代码，用于与预训练 Qwen3 模型交互。

![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)

我们使用开源 [Chainlit Python 包](https://github.com/Chainlit/chainlit) 实现该界面。

&nbsp;
## 步骤 1：安装依赖

首先通过 [requirements-extra.txt](requirements-extra.txt) 安装 `chainlit` 及依赖：

```bash
pip install -r requirements-extra.txt
```

或使用 `uv`：

```bash
uv pip install -r requirements-extra.txt
```

&nbsp;
## 步骤 2：运行 `app` 代码

本文件夹包含 2 个文件：

1. [`qwen3-chat-interface.py`](qwen3-chat-interface.py)：加载并使用 thinking 模式的 Qwen3 0.6B 模型。
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py)：与上面相同，但配置为记住对话历史。

（打开并查看这些文件以了解更多细节。）

在终端中运行以下命令之一以启动 UI 服务：

```bash
chainlit run qwen3-chat-interface.py
```

或使用 `uv`：

```bash
uv run chainlit run qwen3-chat-interface.py
```

运行上述命令后应会自动打开新的浏览器标签页，可在其中与模型交互。若未自动打开，请查看终端输出，将本地地址复制到浏览器地址栏（通常为 `http://localhost:8000`）。
