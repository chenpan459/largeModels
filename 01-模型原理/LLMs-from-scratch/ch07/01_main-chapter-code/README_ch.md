# 第 7 章：微调以遵循指令

### 主章节代码

- [ch07.ipynb](ch07.ipynb) 包含章节中出现的全部代码
- [previous_chapters.py](previous_chapters.py) 是 Python 模块，包含前几章实现并训练的 GPT 模型及诸多工具函数，本章复用
- [gpt_download.py](gpt_download.py) 包含下载预训练 GPT 模型权重的工具函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) / [exercise-solutions_ch.ipynb](exercise-solutions_ch.ipynb) 包含本章练习解答


### 可选代码

- [load-finetuned-model.ipynb](load-finetuned-model.ipynb) / [load-finetuned-model_ch.ipynb](load-finetuned-model_ch.ipynb) 是独立 Jupyter notebook，用于加载本章创建的指令微调模型

- [gpt_instruction_finetuning.py](gpt_instruction_finetuning.py) 是独立 Python 脚本，汇总主章节中的指令微调流程（可视为聚焦微调部分的章节小结）

用法：

```bash
python gpt_instruction_finetuning.py
```

```
matplotlib version: 3.9.0
tiktoken version: 0.7.0
torch version: 2.3.1
tqdm version: 4.66.4
tensorflow version: 2.16.1
--------------------------------------------------
Training set length: 935
Validation set length: 55
Test set length: 110
--------------------------------------------------
Device: cpu
--------------------------------------------------
File already exists and is up-to-date: gpt2/355M/checkpoint
File already exists and is up-to-date: gpt2/355M/encoder.json
File already exists and is up-to-date: gpt2/355M/hparams.json
File already exists and is up-to-date: gpt2/355M/model.ckpt.data-00000-of-00001
File already exists and is up-to-date: gpt2/355M/model.ckpt.index
File already exists and is up-to-date: gpt2/355M/model.ckpt.meta
File already exists and is up-to-date: gpt2/355M/vocab.bpe
Loaded model: gpt2-medium (355M)
--------------------------------------------------
Initial losses
   Training loss: 3.839039182662964
   Validation loss: 3.7619192123413088
Ep 1 (Step 000000): Train loss 2.611, Val loss 2.668
Ep 1 (Step 000005): Train loss 1.161, Val loss 1.131
Ep 1 (Step 000010): Train loss 0.939, Val loss 0.973
...
Training completed in 15.66 minutes.
Plot saved as loss-plot-standalone.pdf
--------------------------------------------------
Generating responses
100%|█████████████████████████████████████████████████████████| 110/110 [06:57<00:00,  3.80s/it]
Responses saved as instruction-data-with-response-standalone.json
Model saved as gpt2-medium355M-sft-standalone.pth
```

- [ollama_evaluate.py](ollama_evaluate.py) 是独立 Python 脚本，汇总主章节中的评估流程（可视为聚焦评估部分的章节小结）

用法：

```bash
python ollama_evaluate.py --file_path instruction-data-with-response-standalone.json
```

```
Ollama running: True
Scoring entries: 100%|███████████████████████████████████████| 110/110 [01:08<00:00,  1.62it/s]
Number of scores: 110 of 110
Average score: 51.75
```

- [exercise_experiments.py](exercise_experiments.py) 是实现练习解答的可选脚本；详见 [exercise-solutions.ipynb](exercise-solutions.ipynb) / [exercise-solutions_ch.ipynb](exercise-solutions_ch.ipynb)


## 中文文档

| 原文 | 中文版 |
|------|--------|
| [README.md](README.md) | [README_ch.md](README_ch.md) |
