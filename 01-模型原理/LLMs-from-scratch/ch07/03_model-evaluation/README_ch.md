# 第 7 章：微调以遵循指令

This folder 包含可用于模型评估的工具代码.



&nbsp;
## 使用 OpenAI API 评估指令回复


- [llm-instruction-eval-openai.ipynb](llm-instruction-eval-openai.ipynb) 使用 OpenAI GPT-4 评估指令微调模型生成的回复。它使用如下格式的 JSON 文件：

```python
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- The target given in the test set
    "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- Response by an LLM
    "model 2 response": "\nThe atomic number of helium is 3."    # <-- Response by a 2nd LLM
},
```

&nbsp;
## 使用 Ollama 在本地评估指令回复

- [llm-instruction-eval-ollama.ipynb](llm-instruction-eval-ollama.ipynb) 是上述 notebook 的替代方案，通过 Ollama 使用本地下载的 Llama 3 模型。

## 中文文档

| 原文 | 中文版 |
|------|--------|
| [README.md](README.md) | [README_ch.md](README_ch.md) |
