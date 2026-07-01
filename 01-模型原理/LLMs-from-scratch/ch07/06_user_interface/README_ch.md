# 构建与指令微调 GPT 模型交互的用户界面



本补充文件夹包含运行 ChatGPT 式用户界面、与第 7 章指令微调 GPT 交互的代码，如下所示。



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-sft.webp?2)



我们使用开源 [Chainlit Python 包](https://github.com/Chainlit/chainlit) 实现该用户界面。

&nbsp;
## 步骤 1：安装依赖

首先，通过以下命令安装 `chainlit` 包：

```bash
pip install chainlit
```

（也可执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 步骤 2：运行 `app` 代码

[`app.py`](app.py) 包含 UI 代码。打开并查看这些文件以了解更多。

该文件加载并使用第 7 章生成的 GPT-2 权重。需先运行 [`../01_main-chapter-code/ch07.ipynb`](../01_main-chapter-code/ch07.ipynb) / [`ch07_ch.ipynb`](ch07_ch.ipynb)。

在终端执行以下命令启动 UI 服务器：

```bash
chainlit run app.py
```

运行上述命令应会打开新的浏览器标签页，可在其中与模型交互。若未自动打开，请查看终端输出并将本地地址复制到浏览器（通常为 `http://localhost:8000`）。

## 中文文档

| 原文 | 中文版 |
|------|--------|
| [README.md](README.md) | [README_ch.md](README_ch.md) |
