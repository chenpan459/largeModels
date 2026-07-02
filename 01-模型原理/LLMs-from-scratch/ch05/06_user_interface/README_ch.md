# 构建与预训练 LLM 交互的用户界面

本 bonus 文件夹包含运行类 ChatGPT 用户界面的代码，用于与第 5 章的预训练 LLM 交互，效果如下所示。

![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-orig.webp)

我们使用开源 [Chainlit Python 包](https://github.com/Chainlit/chainlit) 实现该界面。

&nbsp;
## 步骤 1：安装依赖

首先安装 `chainlit` 包：

```bash
pip install chainlit
```

（也可执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 步骤 2：运行 `app` 代码

本文件夹包含 2 个文件：

1. [`app_orig.py`](app_orig.py)：加载并使用 OpenAI 原始 GPT-2 权重。
2. [`app_own.py`](app_own.py)：加载并使用第 5 章生成的 GPT-2 权重。需先运行 [`../01_main-chapter-code/ch05_ch.ipynb`](../01_main-chapter-code/ch05_ch.ipynb) 生成 `model.pth`。

（打开并查看这些文件以了解更多细节。）

在终端中运行以下命令之一以启动 UI 服务：

```bash
chainlit run app_orig.py
```

或

```bash
chainlit run app_own.py
```

运行上述命令后应会自动打开新的浏览器标签页，可在其中与模型交互。若未自动打开，请查看终端输出，将本地地址复制到浏览器地址栏（通常为 `http://localhost:8000`）。
