# 第 7 章：微调以遵循指令

本文件夹包含可用于模型评估的工具代码。

&nbsp;
## 使用 OpenAI API 评估指令回复

- [llm-instruction-eval-openai.ipynb](llm-instruction-eval-openai.ipynb) / [llm-instruction-eval-openai_ch.ipynb](llm-instruction-eval-openai_ch.ipynb) 使用 OpenAI GPT-4 评估指令微调模型生成的回复。它使用如下格式的 JSON 文件：

```python
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- 测试集中的目标答案
    "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- LLM 的回复
    "model 2 response": "\nThe atomic number of helium is 3."    # <-- 第二个 LLM 的回复
},
```

&nbsp;
## 使用 Ollama 在本地评估指令回复

- [llm-instruction-eval-ollama.ipynb](llm-instruction-eval-ollama.ipynb) / [llm-instruction-eval-ollama_ch.ipynb](llm-instruction-eval-ollama_ch.ipynb) 是上述 notebook 的替代方案，通过 Ollama 使用本地下载的 Llama 3 模型。

- [scores/correlation-analysis.ipynb](scores/correlation-analysis.ipynb) / [scores/correlation-analysis_ch.ipynb](scores/correlation-analysis_ch.ipynb) 分析 GPT-4 与 Llama 3 评分之间的相关性
