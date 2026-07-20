# 06 — 数据模块

## 模块概览

数据模块位于 `src/llamafactory/data/`，负责从原始数据到模型可用 batch 的全流程处理。

```
dataset_info.json → loader.py → converter.py → processor/ → collator.py → Trainer
                         ↑
                    template.py（提示模板）
                         ↑
                    mm_plugin.py（多模态）
```

## 数据集加载

### 统一入口

`data/loader.py` 的 `get_dataset()` 是数据加载的唯一入口：

```python
dataset_module = get_dataset(
    template, model_args, data_args, training_args,
    stage="sft",  # pt / sft / rm / ppo / dpo / kto
    **tokenizer_module
)
# 返回: {"train_dataset", "eval_dataset", "predict_dataset"}
```

### 加载流程

```
get_dataset_list()          # 从 dataset_info.json 解析数据集列表
    ↓
_load_single_dataset()      # 按来源加载单个数据集
    ↓
align_dataset()             # 转换为标准 schema
    ↓
merge_dataset()             # 合并多个数据集
    ↓
split_dataset()             # 划分 train/val
    ↓
DatasetProcessor            # 按 stage tokenize
    ↓
返回 DatasetModule
```

### 数据来源

| 来源 | `load_from` | 说明 |
|------|-------------|------|
| HuggingFace Hub | `hf_hub` | `load_dataset()` |
| ModelScope | `ms_hub` | `MsDataset.load()` |
| OpenMind | `om_hub` | `OmDataset.load_dataset()` |
| 本地文件 | `file` | JSON/JSONL/CSV/Parquet/TXT |
| 本地脚本 | `script` | 自定义 loading script |
| 云端 JSON | `cloud_file` | HTTP URL 读取 |

支持的本地文件格式：`json`、`jsonl`、`csv`、`parquet`、`txt`。

## 提示模板

`data/template.py` 中的 `TEMPLATES` 字典定义了各模型族的对话格式：

```python
TEMPLATES = {
    "qwen3": Template(...),
    "qwen3_nothink": Template(...),
    "llama3": Template(...),
    "chatglm3": Template(...),
    ...
}
```

每个 `Template` 包含：

| 属性 | 说明 |
|------|------|
| `format_user` | 用户消息格式 |
| `format_assistant` | 助手消息格式 |
| `format_system` | 系统消息格式 |
| `format_tools` | 工具调用格式 |
| `format_observation` | 工具返回格式 |
| `stop_words` | 停止词 |
| `efficient_eos` | 高效 EOS 处理 |
| `replace_eos` | 替换 EOS token |

`get_template_and_fix_tokenizer()` 根据 `data_args.template` 选择模板，并修复 tokenizer 的特殊 token（pad、eos 等）。

## 数据处理器

`data/processor/` 目录下按训练 stage 分派不同的 tokenize 逻辑：

| 处理器 | 文件 | 适用 stage | 说明 |
|--------|------|-----------|------|
| `PretrainDatasetProcessor` | `pretrain.py` | pt | 纯文本 CLM |
| `SupervisedDatasetProcessor` | `supervised.py` | sft | 指令-回复对 |
| `PackedSupervisedDatasetProcessor` | `supervised.py` | sft | 序列打包 SFT |
| `PairwiseDatasetProcessor` | `pairwise.py` | rm, dpo | 偏好对（chosen/rejected） |
| `FeedbackDatasetProcessor` | `feedback.py` | kto | 二元反馈 |
| `UnsupervisedDatasetProcessor` | `unsupervised.py` | ppo | 无监督生成 |

### SFT 数据处理示例

输入（Alpaca 格式）：

```json
{
  "instruction": "解释什么是机器学习",
  "input": "",
  "output": "机器学习是人工智能的一个分支..."
}
```

经过 Template + Processor 后变为：

```
input_ids:  [system_tokens, user_tokens, assistant_tokens]
labels:     [-100, -100, ..., assistant_tokens]   # -100 = 不计算 loss
attention_mask: [1, 1, 1, ...]
```

只有 assistant 部分的 token 参与 loss 计算（`IGNORE_INDEX = -100`）。

## 数据格式

### Alpaca 格式

```json
{
  "instruction": "任务描述",
  "input": "额外输入（可选）",
  "output": "期望输出"
}
```

### ShareGPT 格式

```json
{
  "conversations": [
    {"from": "human", "value": "你好"},
    {"from": "gpt", "value": "你好！有什么可以帮助你的？"}
  ]
}
```

ShareGPT 格式支持多轮对话、工具调用和多模态字段（images、videos、audios）。

## 多模态支持

`data/mm_plugin.py` 处理图像、视频、音频输入：

| 模态 | 占位符 | 处理方式 |
|------|--------|---------|
| 图像 | `<image>` | 通过 model processor 编码 |
| 视频 | `<video>` | 帧提取 + 编码 |
| 音频 | `<audio>` | 音频特征提取 |

多模态数据集在 `dataset_info.json` 中通过 `columns` 映射媒体字段：

```json
"mllm_demo": {
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "images": "images"
  }
}
```

## 批处理 Collator

`data/collator.py` 提供 stage 专用的 batch 组装：

| Collator | 说明 |
|----------|------|
| `SFTDataCollatorWith4DAttentionMask` | SFT 训练，支持 4D attention mask（序列打包） |
| `PairwiseDataCollatorWithPadding` | DPO/RM 偏好对 |
| `KTODataCollatorWithPadding` | KTO 二元反馈 |

Collator 负责：padding 对齐、attention mask 构建、labels 填充（pad 位置设为 -100）。

## 自定义数据集

1. 准备 JSON/JSONL 文件，放入 `data/` 目录
2. 在 `data/dataset_info.json` 中注册：

```json
"my_dataset": {
  "file_name": "my_dataset.json",
  "formatting": "alpaca",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

3. 在 YAML 配置中引用：

```yaml
dataset: my_dataset
template: qwen3_nothink
```

## 关键函数速查

| 函数 | 文件 | 说明 |
|------|------|------|
| `get_dataset()` | `loader.py` | 数据集加载总入口 |
| `get_template_and_fix_tokenizer()` | `template.py` | 模板选择 + tokenizer 修复 |
| `get_dataset_list()` | `parser.py` | 解析 dataset_info.json |
| `align_dataset()` | `converter.py` | 原始数据 → 标准 schema |
| `merge_dataset()` | `data_utils.py` | 合并多个数据集 |
| `split_dataset()` | `data_utils.py` | 划分 train/val |
