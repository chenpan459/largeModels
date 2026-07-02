#!/usr/bin/env python3
"""Generate reflection-gpt4_ch.ipynb from reflection-gpt4.ipynb."""
import json
import copy
from pathlib import Path

SRC = Path(__file__).parent / "reflection-gpt4.ipynb"
DST = Path(__file__).parent / "reflection-gpt4_ch.ipynb"

HEADER = """<table style="width:100%">
<tr>
<td style="vertical-align:middle; text-align:left;">
<font size="2">
<a href="http://mng.bz/orYv">《从零构建大语言模型》（Build a Large Language Model From Scratch）</a> 一书的配套代码，作者 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>
<br>代码仓库：<a href="https://github.com/rasbt/LLMs-from-scratch">https://github.com/rasbt/LLMs-from-scratch</a>
</font>
</td>
<td style="vertical-align:middle; text-align:left;">
<a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>
</td>
</tr>
</table>"""

MARKDOWN_TRANSLATIONS = {
    1: "# 使用 GPT-4 实现 Reflection-Tuning 数据集改进",
    2: (
        "- 本 notebook 使用 OpenAI 的 GPT-4 API 实现 [Reflection-Tuning: Data Recycling Improves LLM Instruction-Tuning](https://arxiv.org/abs/2310.11716) 论文中的数据集改进流程\n"
        "\n"
        "![](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/reflection-tuning/reflection-tuning.webp)\n"
        "\n"
        "- 在原论文中，研究人员改进了 [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) 与 [WizardLM](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_70k) 指令微调数据集；在本 notebook 中，我们改进 [第 7 章使用的指令数据集](../01_main-chapter-code/ch07_ch.ipynb)（`instruction-data.json`；因其格式与 Alpaca 相同，相同代码也适用于 Alpaca 数据集）\n"
        "\n"
        "- 预期的数据集格式如下：\n"
        "\n"
        "```python\n"
        "    {\n"
        "        \"instruction\": \"Edit the following sentence for grammar.\",\n"
        "        \"input\": \"He go to the park every day.\",\n"
        "        \"output\": \"He goes to the park every day.\"\n"
        "    },\n"
        "    {\n"
        "        \"instruction\": \"Convert 45 kilometers to meters.\",\n"
        "        \"input\": \"\",\n"
        "        \"output\": \"45 kilometers is 45000 meters.\"\n"
        "    },\n"
        "```"
    ),
    3: (
        "> 请注意，本 notebook 复现了论文中使用 GPT API 增强现有数据集的方法。但需知，根据 [OpenAI 使用条款](https://openai.com/policies/row-terms-of-use/)，GPT API 生成的数据不得用于开发与 OpenAI 竞争的模型：\"What you cannot do... Use Output to develop models that compete with OpenAI.\"\n"
        "相关讨论见 [此处](https://www.reddit.com/r/LocalLLaMA/comments/17vbg1f/does_openai_tos_prohibit_generating_datasets_for/))。"
    ),
    6: "## 测试 OpenAI API",
    7: (
        "- 首先，让我们测试 OpenAI API 是否已正确配置\n"
        "- 若尚无账户，需在 https://platform.openai.com/ 注册\n"
        "- 注意，还需向账户充值，GPT-4 API 并非免费（见 https://platform.openai.com/settings/organization/billing/overview）\n"
        "- 按本 notebook 原样运行代码，使用 GPT-4o-mini 时费用约为 \\$0.03（3 美分）\n"
        "- 将上述两种方法应用于第 7 章指令数据集中全部 1100 条记录，费用约为 \\$0.60（60 美分）"
    ),
    8: (
        "- 首先，我们需要提供 OpenAI API 密钥，可在 https://platform.openai.com/api-keys 获取\n"
        "- 切勿与他人分享此密钥\n"
        "- 请将此密钥（`\"sk-...\"`）添加到本文件夹中的 `config.json` 文件"
    ),
    10: "- 首先，让我们用一个简单示例测试 API 是否按预期工作：",
    12: "## 加载 JSON 条目",
    13: (
        "- 接下来，加载并处理指令数据集\n"
        "- 此处假设我们已将测试数据集与模型回复保存为 JSON 文件，可按如下方式加载："
    ),
    15: "- 打印一条数据集记录以查看其结构：",
    17: "## 改进指令",
    18: (
        "- Reflection-Tuning 作者分享了两种方法：（1）改进指令，（2）改进回复\n"
        "- 我们先从改进数据集中的指令开始\n"
        "- 下面是 [Reflection-Tuning 仓库](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) 中用于格式化 GPT-4 模型输入以进行数据集改进的小型工具函数"
    ),
    20: "- 为演示其工作原理，考虑数据集条目 `json_data[2]`",
    22: "- 可使用上文定义的 `build_instruction_reflection_prompt_no_input` 函数改进指令：",
    24: (
        "- 回复非常冗长，这对分析很有用；也有助于 GPT-4 模型通过思维链（chain-of-thought）提示进行改进\n"
        "- 但为构建改进后的数据集，我们实际上只关心新指令与输出，而非分析内容\n"
        "- 可使用 [Reflection-Tuning 仓库](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) 中的以下工具代码，从 GPT-4 输出中提取改进后的指令与回复"
    ),
    26: "- 使用这些工具函数，从先前生成的冗长 GPT-4 输出中提取改进后的指令与回复：",
    30: '- 请注意，指令改进目前仅针对不含 `"input"` 字段的数据集条目实现',
    31: "## 改进回复",
    32: (
        "- 类似地，我们也可将 Reflection-Tuning 改进流程专门应用于数据集的回复（即 \"output\" 字段）\n"
        "- 下面是 [Reflection-Tuning 仓库](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) 中用于格式化 GPT-4 模型输入以进行数据集改进的两个小型工具函数"
    ),
    34: "- 再次对一条数据集条目应用，查看如何生成改进后的回复：",
    36: (
        "- 如上所示，回复包含对原始回复的分析；可使用 [Reflection-Tuning 仓库](https://github.com/tianyi-lab/Reflection_Tuning/blob/main/reflection_code/reflect_response.py) 中的以下工具函数提取新回复"
    ),
    39: "## 改进数据集",
    40: (
        "- 现在，将指令反思与回复反思技术应用于实际数据集\n"
        "- 注意：此处为演示目的仅处理一小部分数据；要对整个数据集应用，请将\n"
        "\n"
        "```python\n"
        "data_to_process = json_data[:3]\n"
        "```\n"
        "\n"
        "改为\n"
        "\n"
        "```python\n"
        "data_to_process = json_data\n"
        "```"
    ),
    41: "### 反思指令",
    42: "- 以下代码将 Reflection-Tuning 数据集改进方法应用于原始数据集中的指令",
    47: "- 保存新数据集：",
    49: "### 反思回复",
    50: "- 现在对回复反思执行相同操作：",
    55: "- 保存新数据集：",
    57: "## 创建改进后的指令数据",
    58: (
        "- 将上述两种方法应用于第 7 章指令数据集中全部 1100 条记录，费用约为 \\$0.60（60 美分）\n"
        "- 为避免 GitHub 仓库因数据集文件而膨胀，生成的数据集文件可从 Google Drive 获取：\n"
        "  - [instruction-reflected.json](https://drive.google.com/file/d/1c1QnuTdt9nP1u51vBn4_b05mWR_ZNGBv/view?usp=sharing)\n"
        "  - [response-reflected.json](https://drive.google.com/file/d/1RNckTZ2ELcdUoJtaylao6NvyZPMtNv1v/view?usp=sharing)"
    ),
}

CODE_TRANSLATIONS = {
    5: (
        "from importlib.metadata import version\n"
        "\n"
        "pkgs = [\n"
        "    \"openai\",  # OpenAI API\n"
        "    \"tqdm\",    # 进度条\n"
        "]\n"
        "\n"
        "for p in pkgs:\n"
        "    print(f\"{p} 版本: {version(p)}\")"
    ),
    9: (
        "import json\n"
        "from openai import OpenAI\n"
        "\n"
        "# 从 JSON 文件加载 API 密钥。\n"
        "# 请确保将 \"sk-...\" 替换为你在 https://platform.openai.com/api-keys 获取的实际 API 密钥\n"
        "with open(\"config.json\", \"r\") as config_file:\n"
        "    config = json.load(config_file)\n"
        "    api_key = config[\"OPENAI_API_KEY\"]\n"
        "\n"
        "client = OpenAI(api_key=api_key)"
    ),
    11: (
        "def run_chatgpt(prompt, client, model=\"gpt-4o-mini\", system_prompt=None):\n"
        "    # 若提供了 system_prompt，则定义 system 消息\n"
        "    messages = []\n"
        "    \n"
        "    if system_prompt:\n"
        "        messages.append({\"role\": \"system\", \"content\": system_prompt})\n"
        "    \n"
        "    # 将用户 prompt 添加到 messages\n"
        "    messages.append({\"role\": \"user\", \"content\": prompt})\n"
        "\n"
        "    # 调用 API\n"
        "    response = client.chat.completions.create(\n"
        "        model=model,\n"
        "        messages=messages,\n"
        "        temperature=0.0,\n"
        "        seed=123,\n"
        "    )\n"
        "    \n"
        "    # 返回模型回复\n"
        "    return response.choices[0].message.content\n"
        "\n"
        "\n"
        "prompt = \"Respond with 'hello world' if you got this message.\"\n"
        "run_chatgpt(prompt, client)"
    ),
    14: (
        "from pathlib import Path\n"
        "\n"
        "\n"
        "json_file = Path(\"..\") / \"01_main-chapter-code\" / \"instruction-data.json\"\n"
        "\n"
        "with open(json_file, \"r\") as file:\n"
        "    json_data = json.load(file)\n"
        "\n"
        "print(\"条目数量:\", len(json_data))"
    ),
}


def to_source(text: str) -> list[str]:
    lines = text.split("\n")
    if not lines:
        return []
    result = [lines[0] + "\n"]
    for line in lines[1:]:
        result.append(line + "\n")
    if not text.endswith("\n") and result:
        result[-1] = result[-1].rstrip("\n")
    return result


def clear_cell(cell: dict) -> None:
    cell["execution_count"] = None
    cell["outputs"] = []


def main() -> None:
    with open(SRC, encoding="utf-8") as f:
        nb = json.load(f)

    out = copy.deepcopy(nb)
    assert len(out["cells"]) == 59, f"Expected 59 cells, got {len(out['cells'])}"

    for i, cell in enumerate(out["cells"]):
        if cell["cell_type"] == "code":
            clear_cell(cell)
            if i in CODE_TRANSLATIONS:
                cell["source"] = to_source(CODE_TRANSLATIONS[i])
        elif cell["cell_type"] == "markdown":
            if i == 0:
                cell["source"] = to_source(HEADER)
            elif i in MARKDOWN_TRANSLATIONS:
                cell["source"] = to_source(MARKDOWN_TRANSLATIONS[i])

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"Wrote {DST} with {len(out['cells'])} cells")


if __name__ == "__main__":
    main()
