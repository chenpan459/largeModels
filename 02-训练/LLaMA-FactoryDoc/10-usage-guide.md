# 10 — 使用指南

## 安装

环境搭建（Python 3.11、虚拟环境、依赖安装、排错）见 **[00-environment-setup.md](./00-environment-setup.md)**。

前置条件：已激活虚拟环境并验证 `llamafactory-cli version` 正常。

## 快速开始：LoRA 微调

### 1. 准备数据

使用内置演示数据集，或添加自定义数据：

```bash
# 查看可用数据集
cat data/dataset_info.json | python -m json.tool | head -20
```

自定义数据集步骤见 [06-data-module.md](./06-data-module.md)。

### 2. 选择配置

```bash
# 查看 SFT 配置
cat examples/train_lora/qwen3_lora_sft.yaml
```

### 3. 启动训练

```bash
# 单卡
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# 多卡（自动 torchrun）
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# 覆盖参数
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml \
  learning_rate=5e-5 num_train_epochs=1 max_samples=100
```

### 4. 监控训练

- 终端日志：loss、learning_rate、进度
- Loss 曲线：`output_dir/training_loss.png`（需 `plot_loss: true`）
- W&B：设置 `report_to: wandb`

### 5. 测试模型

```bash
# CLI 对话
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml

# 或启动 Web UI
llamafactory-cli webui
```

## 常见训练场景

### SFT 监督微调

```yaml
stage: sft
finetuning_type: lora
lora_rank: 8
dataset: alpaca_en_demo
template: qwen3_nothink
```

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

### QLoRA 量化微调

显存不足时使用 4-bit 量化：

```yaml
finetuning_type: lora
quantization_bit: 4
quantization_method: bitsandbytes
```

```bash
llamafactory-cli train examples/train_qlora/qwen3_lora_sft.yaml
```

### DPO 偏好对齐

```yaml
stage: dpo
pref_beta: 0.1
pref_loss: sigmoid  # sigmoid / orpo / simpo
dataset: dpo_en_demo
```

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml
```

### 继续预训练

```yaml
stage: pt
dataset: c4_demo
```

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_pretrain.yaml
```

### 全量微调

```yaml
finetuning_type: full
bf16: true
deepspeed: examples/deepspeed/ds_z3_config.json
```

```bash
llamafactory-cli train examples/train_full/qwen3_full_sft.yaml
```

### 多模态微调

```yaml
stage: sft
dataset: mllm_demo
template: qwen3_vl
```

```bash
llamafactory-cli train examples/train_lora/qwen3vl_lora_sft.yaml
```

## 模型导出

### 合并 LoRA

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
adapter_name_or_path: saves/qwen3-4b/lora/sft
template: qwen3_nothink
export_dir: exports/qwen3-4b-merged
export_size: 2
export_device: cpu
```

```bash
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

导出后在 `exports/qwen3-4b-merged/` 获得完整 HuggingFace 格式模型。

## API 部署

### 启动服务

```bash
# 使用微调后的 LoRA
llamafactory-cli api examples/inference/qwen3_lora_sft.yaml

# 或设置环境变量
API_HOST=0.0.0.0 API_PORT=8000 API_KEY=sk-xxx llamafactory-cli api
```

### 调用示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-xxx")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

### vLLM 加速部署

```yaml
infer_backend: vllm
vllm_enforce_eager: true
```

## 分布式训练

### 多 GPU（自动）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config.yaml
# launcher.py 自动检测 4 卡并启动 torchrun
```

### DeepSpeed ZeRO-3

```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
```

### 多节点

```bash
# 节点 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 \
  llamafactory-cli train config.yaml

# 节点 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 \
  llamafactory-cli train config.yaml
```

## Web UI 操作

```bash
llamafactory-cli webui
```

浏览器打开 `http://localhost:7860`：

1. **Train** Tab：选择模型、数据集、训练参数，点击 Start
2. **Chat** Tab：加载模型测试对话效果
3. **Export** Tab：合并 LoRA 导出

## 断点续训

```yaml
resume_from_checkpoint: saves/qwen3-4b/lora/sft/checkpoint-500
```

或在 Web UI 的 Train Tab 中选择 checkpoint 路径。

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| CUDA OOM | 减小 batch_size、启用 QLoRA、使用 DeepSpeed ZeRO-3 |
| 模型下载慢 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 或使用 ModelScope |
| LoRA 合并失败 | 确保 `adapter_name_or_path` 指向正确的 checkpoint 目录 |
| 多卡不生效 | 检查 `CUDA_VISIBLE_DEVICES`，确认 `get_device_count() > 1` |
| 模板不匹配 | 确认 `template` 与模型匹配（如 Qwen3 用 `qwen3_nothink`） |

## 工具脚本

| 脚本 | 说明 |
|------|------|
| `scripts/vllm_infer.py` | vLLM 批量推理 |
| `scripts/eval_bleu_rouge.py` | BLEU/ROUGE 评测 |
| `scripts/convert_ckpt/` | 检查点格式转换 |
| `scripts/stat_utils/` | 数据集统计 |
