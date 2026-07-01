# 第 7 章：微调以遵循指令

本文件夹包含可用于准备指令数据集的工具代码。

通过以下命令安装额外依赖：

```bash
pip install -r requirements-extra.txt
```

### 查找近似重复

`find-near-duplicates.py` 可用于识别指令数据集中的重复与近似重复条目。例如，

```bash
python find-near-duplicates.py --json_file instruction-examples.json
```

```
scikit-learn version: 1.3.1


==================================================
Searching 'instruction' for duplicates ...
==================================================
Duplicate pair found with similarity 0.94:
1. Edit the following sentence to make it more formal.
2. Edit the sentence to make it more formal.

Duplicate pair found with similarity 1.00:
1. Name a dwarf planet in our solar system.
2. Name a dwarf planet in our solar system.

Duplicate pair found with similarity 0.91:
1. Change the sentences from active voice to passive voice.
2. Change the sentence from passive to active voice.



==================================================
Searching 'input' for duplicates ...
==================================================
No duplicates found


==================================================
Searching 'output' for duplicates ...
==================================================
Duplicate pair found with similarity 1.00:
1. One dwarf planet in our solar system is Pluto.
2. One dwarf planet in our solar system is Pluto.


```

&nbsp;
可使用 0 到 1 之间的 `--threshold` 降低或提高敏感度。
默认阈值为 0.9。

&nbsp;
## 创建被动语态条目

- [create-passive-voice-entries.ipynb](create-passive-voice-entries.ipynb) / [create-passive-voice-entries_ch.ipynb](create-passive-voice-entries_ch.ipynb) notebook 使用 OpenAI GPT-4 为指令数据集创建「被动语态」条目，示例如下

```python
{  
   'instruction': 'Identify the verb in the following sentence',
   'input': 'The cat sleeps on the couch.',
   'output': 'The verb in the sentence is "sleeps."',
   'output_2': 'The sentence is "sleeps."'   #  <---- 新创建的条目
}  
```
