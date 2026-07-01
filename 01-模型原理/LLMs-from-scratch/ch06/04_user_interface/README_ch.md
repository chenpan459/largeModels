# 构建与基于 GPT 的垃圾短信分类器交互的用户界面



本 bonus 文件夹包含代码，用于运行类似 ChatGPT 的界面，与第 6 章微调后的 GPT 垃圾短信分类器交互，如下所示。



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-spam.webp)



我们使用开源 [Chainlit Python 包](https://github.com/Chainlit/chainlit) 实现该界面。

&nbsp;
## 步骤 1：安装依赖

首先通过以下命令安装 `chainlit` 包：

```bash
pip install chainlit
```

（或执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 步骤 2：运行 `app` 代码

[`app.py`](app.py) 文件包含 UI 代码。打开并查看这些文件以了解更多。

该文件加载并使用我们在第 6 章生成的 GPT-2 分类器权重。需先运行 [`../01_main-chapter-code/ch06.ipynb`](../01_main-chapter-code/ch06.ipynb) / [`ch06_ch.ipynb`](../01_main-chapter-code/ch06_ch.ipynb)。

在终端执行以下命令启动 UI 服务器：

```bash
chainlit run app.py
```

运行上述命令应会打开新浏览器标签页以与模型交互。若未自动打开，请查看终端输出并将本地地址复制到浏览器（通常为 `http://localhost:8000`）。
