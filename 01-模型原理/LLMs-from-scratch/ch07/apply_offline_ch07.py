#!/usr/bin/env python3
"""Apply offline Chinese translations to ch07 *_ch.ipynb markdown cells."""

import json
import re
from pathlib import Path

CH07 = Path(__file__).resolve().parent

# Shared complete Chinese blocks (match partial English in broken cells)
OLLAMA_INTRO = (
    "- Ollama 是一款高效运行 LLM 的应用\n"
    "- 它是 [llama.cpp](https://github.com/ggerganov/llama.cpp) 的封装，后者用纯 C/C++ 实现 LLM 以最大化效率\n"
    "- 注意，它是用于 LLM 文本生成（推理）的工具，而非训练或微调 LLM\n"
    "- 运行下方代码前，请访问 [https://ollama.com](https://ollama.com) 并按说明安装 ollama"
    "（例如点击「Download」按钮，下载适用于您操作系统的 ollama 应用）"
)

OLLAMA_REST = (
    "- 现在，与模型交互的另一种方式是通过 Python 调用其 REST API，使用以下函数\n"
    "- 运行本 notebook 后续单元格前，请确保 ollama 仍在运行，方式同上：\n"
    "  - 在终端中运行 `ollama serve`\n"
    "  - 或启动 ollama 应用\n"
    "- 接下来，运行以下代码单元格查询模型"
)

OPENAI_TEST = (
    "- 首先，让我们测试 OpenAI API 是否已正确配置\n"
    "- 若尚无账户，需在 https://platform.openai.com/ 注册\n"
    "- 注意，还需向账户充值，GPT-4 API 并非免费"
    "（见 https://platform.openai.com/settings/organization/billing/overview）"
)

OPENAI_KEY = (
    "- 首先，需提供 OpenAI API 密钥，可在 https://platform.openai.com/api-keys 获取\n"
    "- 切勿与他人分享此密钥\n"
    "- 将该密钥（`\"sk-...\"`）添加到本文件夹的 `config.json` 文件中"
)

# Full EN -> ZH replacements (longest first)
REPLACEMENTS = [
    ("- We tackle this dataset batching in several steps, as summarized in the figure below",
     "- 我们分多个步骤处理数据集批处理，如下图所示"),
    ("- First, we implement an `InstructionDataset` class that pre-tokenizes all inputs in the dataset, similar to the `SpamDataset` in chapter 6",
     "- 首先，我们实现 `InstructionDataset` 类，对数据集中所有输入进行预分词，类似于第 6 章的 `SpamDataset`"),
    ("- Similar to chapter 6, we want to collect multiple training examples in a batch to accelerate training; this requires padding all inputs to a similar length",
     "- 与第 6 章类似，我们希望在批次中收集多个训练样本以加速训练；这要求将所有输入填充到相近长度"),
    ("- Also similar to the previous chapter, we use the `<|endoftext|>` token as a padding token",
     "- 同样与上一章类似，我们使用 `<|endoftext|>` token 作为填充 token"),
    ("- In chapter 6, we padded all examples in a dataset to the same length",
     "- 在第 6 章中，我们将数据集中所有样本填充到相同长度"),
    ("  - Here, we take a more sophisticated approach and develop a custom \"collate\" function that we can pass to the data loader",
     "  - 此处我们采用更精细的方法，开发可传入数据加载器的自定义 collate 函数"),
    ("  - This custom collate function pads the training examples in each batch to have the same length (but different batches can have different lengths)",
     "  - 该自定义 collate 函数将每批训练样本填充到相同长度（但不同批次可有不同长度）"),
    ("- Above, we only returned the inputs to the LLM; however, for LLM training, we also need the target values",
     "- 上文仅返回了 LLM 的输入；但 LLM 训练还需要目标值"),
    ("- Similar to pretraining an LLM, the targets are the inputs shifted by 1 position to the right, so the LLM learns to predict the next token",
     "- 与预训练 LLM 类似，目标是输入向右移动 1 位的序列，使 LLM 学习预测下一个 token"),
    ("- (In addition, we also introduce the `allowed_max_length` in case we want to limit the length of the samples; this will be useful if you plan to work with your own datasets that are longer than the 1024 token context size supported by the GPT-2 model)",
     "- （此外，我们还引入 `allowed_max_length`，以便限制样本长度；若计划使用超过 GPT-2 模型支持的 1024 token 上下文长度的自定义数据集，这将很有用）"),
    ("- Let's see what this replacement by -100 accomplishes",
     "- 让我们看看用 -100 替换的效果"),
    ("- As we can see, the resulting loss on these 3 training examples is the same as the loss we calculated from the 2 training examples, which means that the cross-entropy loss function ignored the training example with the -100 label",
     "- 可见，这 3 个训练样本的损失与 2 个训练样本时相同，说明交叉熵损失函数忽略了标签为 -100 的训练样本"),
    ("- In practice, it is also common to mask out the target token IDs that correspond to the instruction, as illustrated in the figure below (this is a recommended reader exercise after completing the chapter)",
     "- 实践中，也常对指令对应的目标 token ID 进行掩码，如下图所示（这是完成本章后推荐的读者练习）"),
    ("- Let's see what the dimensions of the resulting input and target batches look like",
     "- 让我们看看所得输入与目标批次的维度"),
    ("- As we can see based on the output above, all batches have a batch size of 8 but a different length, as expected",
     "- 从上方输出可见，所有批次的 batch size 均为 8，但长度不同，符合预期"),
    ("- Let's also double-check that the inputs contain the `<|endoftext|>` padding tokens corresponding to token ID 50256 by printing the contents of the first training example in the `inputs` batch",
     "- 我们还通过打印 `inputs` 批次中第一个训练样本的内容，确认输入包含对应 token ID 50256 的 `<|endoftext|>` 填充 token"),
    ("- As we can see, the model is not capable of following the instructions, yet; it creates a \"Response\" section but it simply repeats the original input sentence as well as the instruction",
     "- 可见，模型尚不能遵循指令；它虽创建了「Response」部分，但只是重复原始输入句子和指令"),
    ("- Let's calculate the initial training and validation set loss before we start training (as in previous chapters, the goal is to minimize the loss)",
     "- 在开始训练前，让我们计算初始训练集与验证集损失（与前几章一样，目标是最小化损失）"),
    ("- The runtimes for various devices are shown for reference below (running this notebook on a compatible GPU device requires no changes to the code)",
     "- 下方列出了各设备的运行时间供参考（在兼容 GPU 上运行本 notebook 无需修改代码）"),
    ("- As we can see based on the outputs above, the model trains well, as we can tell based on the decreasing training loss and validation loss values",
     "- 从上方输出可见，模型训练良好，训练损失与验证损失均在下降"),
    ("- Furthermore, based on the response text printed after each epoch, we can see that the model correctly follows the instruction to convert the input sentence `'The chef cooks the meal every day.'` into passive voice `'The meal is cooked every day by the chef.'` (We will properly format and evaluate the responses in a later section)",
     "- 此外，从每轮后打印的回复可见，模型正确遵循指令，将输入句 `'The chef cooks the meal every day.'` 转换为被动语态 `'The meal is cooked every day by the chef.'`（我们将在后续章节中正确格式化并评估回复）"),
    ("- As we can see, the loss decreases sharply at the beginning of the first epoch, which means the model starts learning quickly",
     "- 可见，第一轮 epoch 开始时损失急剧下降，说明模型学习很快"),
    ("- We can see that slight overfitting sets in at around 1 training epoch",
     "- 可见，约在 1 个训练 epoch 时出现轻微过拟合"),
    ("- We also save a copy of the model for future use",
     "- 我们还保存模型副本以供将来使用"),
    ("- As we can see based on the test set instructions, given responses, and the model's responses, the model performs relatively well",
     "- 从测试集指令、给定回复与模型回复可见，模型表现相对较好"),
    ("- The answers to the first and last instructions are clearly correct",
     "- 第一和最后一条指令的答案明显正确"),
    ("- The second answer is close; the model answers with \"cumulus cloud\" instead of \"cumulonimbus\" (however, note that cumulus clouds can develop into cumulonimbus clouds, which are capable of producing thunderstorms)",
     "- 第二个答案接近；模型回答「cumulus cloud」而非「cumulonimbus」（不过，积云可发展为积雨云，后者可产生雷暴）"),
    ("- In practice, instruction-finetuned LLMs such as chatbots are evaluated via multiple approaches",
     "- 实践中，指令微调 LLM（如聊天机器人）通过多种方式评估"),
    ("- In the next section, we will use an approach similar to AlpacaEval and use another LLM to evaluate the responses of our model; however, we will use our own test set instead of using a publicly available benchmark dataset",
     "- 下一节中，我们将采用类似 AlpacaEval 的方法，用另一个 LLM 评估模型回复；不过，我们使用自己的测试集，而非公开基准数据集"),
    ("- Let's double-check one of the entries to see whether the responses have been added to the `test_data` dictionary correctly",
     "- 让我们复查一条记录，确认回复已正确添加到 `test_data` 字典"),
    ("- In particular, we use an instruction-finetuned 8-billion-parameter Llama 3 model by Meta AI that can be run locally via ollama ([https://ollama.com](https://ollama.com))",
     "- 具体而言，我们使用 Meta AI 的 80 亿参数指令微调 Llama 3 模型，可通过 ollama ([https://ollama.com](https://ollama.com)) 在本地运行"),
    ("- The following code checks whether the ollama session is running correctly before proceeding to use ollama to evaluate the test set responses we generated in the previous section",
     "- 以下代码检查 ollama 会话是否正常运行，然后再用 ollama 评估上一节生成的测试集回复"),
    ("- As we can see, the Llama 3 model provides a reasonable evaluation and also gives partial points if a model is not entirely correct, as we can see based on the \"cumulus cloud\" answer",
     "- 可见，Llama 3 模型给出了合理评估，对不完全正确的回答也会给部分分数，如「cumulus cloud」答案所示"),
    ("- The evaluation of the 110 entries in the test set takes about 1 minute on an M3 MacBook Air laptop",
     "- 在 M3 MacBook Air 上评估测试集 110 条记录约需 1 分钟"),
    ("- We covered the major steps of the LLM development cycle: implementing an LLM architecture, pretraining an LLM, and finetuning it",
     "- 我们涵盖了 LLM 开发周期的主要步骤：实现 LLM 架构、预训练 LLM 并微调"),
    ("- In my opinion, implementing an LLM from scratch is the best way to understand how LLMs work; I hope you gained a better understanding through this approach",
     "- 在我看来，从零实现 LLM 是理解 LLM 工作原理的最佳方式；希望您通过这种方法有了更深入的理解"),
    ("- The [./load-finetuned-model.ipynb](./load-finetuned-model.ipynb) notebook illustrates how to load the finetuned model in a new session",
     "- [./load-finetuned-model.ipynb](./load-finetuned-model.ipynb) / [./load-finetuned-model_ch.ipynb](./load-finetuned-model_ch.ipynb) notebook 演示如何在新会话中加载微调后的模型"),
    ("- The complete list of bonus materials can be viewed in the main README's [Bonus Material](https://github.com/rasbt/LLMs-from-scratch?tab=readme-ov-file#bonus-material) section",
     "- 完整 bonus 材料列表见主 README 的 [Bonus Material](https://github.com/rasbt/LLMs-from-scratch?tab=readme-ov-file#bonus-material) 部分"),
    ("- Instruction finetuning is often referred to as \"supervised instruction finetuning\" because it involves training a model on a dataset where the input-output pairs are explicitly provided",
     "- 指令微调常被称为「监督式指令微调」，因为它在输入-输出对明确提供的数据集上训练模型"),
    ("- There are different ways to format the entries as inputs to the LLM; the figure below illustrates two example formats that were used for training the Alpaca",
     "- 将条目格式化为 LLM 输入的方式有多种；下图展示了用于训练 Alpaca"),
    ("- Each item in the `data` list we loaded from the JSON file above is a dictionary in the following form",
     "- 从上述 JSON 文件加载的 `data` 列表中，每项为如下形式的字典"),
    ("- Note that the `'input'` field can be empty:",
     "- 注意，`'input'` 字段可以为空："),
    ("- The topics covered in this chapter are summarized in the figure below",
     "- 本章涵盖的主题总结见下图"),
    ("This notebook contains minimal code to load the finetuned model",
     "本 notebook 含最少代码，用于加载微调后的模型"),
    ("``````", "```"),
    ("- Ollama is an application to run LLMs efficiently",
     "- Ollama 是一款高效运行 LLM 的应用"),
    ("- It is a wrapper around [llama.cpp]",
     "- 它是 [llama.cpp]"),
    ("- 注意，it is a tool for using LLMs",
     "- 注意，它是用于 LLM"),
    ("- 注意， it is a tool for using LLMs",
     "- 注意，它是用于 LLM"),
    ("- Prior to running the code below, install ollama",
     "- 运行下方代码前，请安装 ollama"),
    ("- 运行下方代码前，请访问 [https://ollama.com](https://ollama.com) 并按说明安装 ollama (for instance",
     "- 运行下方代码前，请访问 [https://ollama.com](https://ollama.com) 并按说明安装 ollama（例如"),
    ("- Linux 用户可使用官网提供的安装命令 the ollama website",
     "- Linux 用户可使用 ollama 官网提供的安装命令"),
    ("- In general, before we can use ollama from the command line",
     "- 通常，要从命令行使用 ollama"),
    ("- 现在，an alternative way to interact with the model",
     "- 现在，与模型交互的另一种方式"),
    ("- 现在，an alternative way to the `ollama run` command",
     "- 现在，与之前使用的 `ollama run` 命令不同，另一种方式"),
    ("- 在此之前，you run the next cells",
     "- 运行后续单元格前"),
    ("- 在此之前，running the code below, install ollama",
     "- 运行下方代码前，请安装 ollama"),
    ("- 接下来，run the following code cell",
     "- 接下来，运行以下代码单元格"),
    ("- 首先，let's test if the OpenAI API",
     "- 首先，让我们测试 OpenAI API"),
    ("- 如果you don't have an account yet",
     "- 若您尚无账户"),
    ("- 注意，you will also have to transfer",
     "- 注意，您还需向账户充值"),
    ("- 首先，we need to provide our OpenAI API",
     "- 首先，需提供 OpenAI API"),
    ("- 微调后的", "- 微调后的"),
    ("- 预训练", "- 预训练"),
    ("- 训练 ", "- 训练 "),
    ("- 验证 ", "- 验证 "),
    ("- 测试集", "- 测试集"),
    ("- 数据集", "- 数据集"),
    ("- 数据加载器", "- 数据加载器"),
    ("- 指令微调", "- 指令微调"),
    ("load-微调后的-model", "load-finetuned-model"),
    ("instruction-微调后的", "指令微调后的"),
]

# Fix broken partial replacements
PARTIAL_FIXES = [
    (r"指令微调 is often referred to as \"supervised 指令微调\"",
     "指令微调常被称为「监督式指令微调」"),
    (r"involves 训练 a model on a 数据集",
     "在输入-输出对明确提供的数据集上训练模型"),
    (r"used for 训练 the Alpaca",
     "用于训练 Alpaca"),
    (r"PyTorch 数据加载器s", "PyTorch 数据加载器"),
    (r"into a 训练, 验证, and 测试集", "为训练集、验证集和测试集"),
    (r"this 数据集 batching", "该数据集批处理"),
    (r"in the 数据集", "在数据集中"),
    (r"multiple 训练 examples", "多个训练样本"),
    (r"accelerate 训练", "加速训练"),
    (r"for LLM 训练", "对于 LLM 训练"),
    (r"Similar to pre训练", "与预训练"),
    (r"3 训练 examples", "3 个训练样本"),
    (r"2 训练 examples", "2 个训练样本"),
    (r"the 训练 example", "该训练样本"),
    (r"first 训练 example", "第一个训练样本"),
    (r"initial 训练 and 验证", "初始训练与验证"),
    (r"decreasing 训练 loss and 验证 loss", "训练损失与验证损失下降"),
    (r"1 训练 epoch", "1 个训练 epoch"),
    (r"the 测试集", "测试集"),
    (r"pre训练 an LLM, and 微调", "预训练 LLM 并微调"),
    (r"load-微调后的-model", "load-finetuned-model"),
    (r"微调后的 model", "微调后的模型"),
    (r"not 训练 or 微调", "非训练或微调"),
]


def chinese_ratio(text: str) -> float:
    if not text.strip():
        return 1.0
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[a-zA-Z]", text))
    total = cn + en
    return cn / total if total else 1.0


def fix_text(text: str) -> str:
    result = text
    for en, zh in REPLACEMENTS:
        result = result.replace(en, zh)
    for pat, repl in PARTIAL_FIXES:
        result = re.sub(pat, repl, result)
    # Apply shared blocks if cell still looks like ollama intro
    if "llama.cpp" in result and chinese_ratio(result) < 0.6:
        if "wrapper around" in text or "Ollama 是" in result or "Ollama is" in text:
            if "## 安装 Ollama" in text or "ollama.com" in text:
                if "REST API" not in text:
                    return OLLAMA_INTRO if "70-billion" not in text and "70B" not in text else result
    if "REST API in Python" in text or "an alternative way" in text:
        return OLLAMA_REST
    if "test if the OpenAI API" in text or "let's test if the OpenAI" in text:
        return OPENAI_TEST + "\n" + result.split("\n", 1)[-1] if "\n" in result else OPENAI_TEST
    return result


def process_notebook(ch_path: Path) -> int:
    en_path = Path(str(ch_path).replace("_ch.ipynb", ".ipynb"))
    if not en_path.exists():
        return 0
    en_nb = json.loads(en_path.read_text(encoding="utf-8"))
    ch_nb = json.loads(ch_path.read_text(encoding="utf-8"))
    changes = 0
    for en_cell, ch_cell in zip(en_nb["cells"], ch_nb["cells"]):
        if en_cell.get("cell_type") != "markdown":
            continue
        ch_src = "".join(ch_cell.get("source", []))
        new_src = fix_text(ch_src)
        if new_src != ch_src:
            lines = new_src.splitlines(keepends=True)
            ch_cell["source"] = lines if lines else [new_src]
            changes += 1
    if changes:
        ch_path.write_text(json.dumps(ch_nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changes


# Import section headings from translate_ch07
try:
    from translate_ch07 import SECTION_HEADINGS, apply_static_replacements
except ImportError:
    SECTION_HEADINGS = {}
    def apply_static_replacements(t): return t


def main():
    total = 0
    for ch_path in sorted(CH07.rglob("*_ch.ipynb")):
        n = process_notebook(ch_path)
        if n:
            print(f"Fixed {ch_path.relative_to(CH07)}: {n} cells")
            total += n
    print(f"Done: {total} cells fixed")


if __name__ == "__main__":
    main()
