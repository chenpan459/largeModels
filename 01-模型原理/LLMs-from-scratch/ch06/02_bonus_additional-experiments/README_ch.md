# 额外的分类微调实验

下表增加实验以回答有关各种设计选择的额外问题。第一行与主章节设置相同，作为参考。
例如，

- 比较第 1 行与第 2 行回答：「训练最后 token 与第一个 token 时性能有何差异？」；
- 比较第 1 行与第 3 行回答：「仅训练最后一层而非最后一个块时性能有何差异？」；
- 依此类推。

&nbsp;

|      | Model              | Weights    | Trainable token position | Trainable layers | Context length                                         | Training acc | Validation acc | Test acc | Training time | CPU/GPU |
| ---- | ------------------ | ---------- | ------------------------ | ---------------- | ------------------------------------------------------ | ------------ | -------------- | -------- | ------------- | ------- |
| 1    | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 2    | gpt2-small (124M)  | pretrained | first                    | last_block       | longest train ex. (120)                                | 78.46%       | 80.54%         | 75.00%   | 0.28 min      | A100    |
| 3    | gpt2-small (124M)  | pretrained | last                     | last_layer       | longest train ex. (120)                                | 78.65%       | 79.87%         | 72.00%   | 0.25 min      | A100    |
| 4    | gpt2-small (124M)  | pretrained | last                     | last_two_blocks  | longest train ex. (120)                                | 98.85%       | 98.66%         | 98.33%   | 0.33 min      | A100    |
| 5    | gpt2-small (124M)  | pretrained | last                     | all              | longest train ex. (120)                                | 99.62%       | 96.64%         | 96.67%   | 0.69 min      | A100    |
| 6    | gpt2-medium (355M) | pretrained | last                     | last_block       | longest train ex. (120)                                | 87.50%       | 91.28%         | 84.67%   | 0.75 min      | A100    |
| 7    | gpt2-large (774M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.52%       | 98.66%         | 96.67%   | 1.50 min      | A100    |
| 8    | gpt2-xl (1558M)    | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.81%       | 99.81%         | 98.33%   | 2.83 min      | A100    |
| 9    | gpt2-xl (1558M)    | pretrained | last                     | all              | longest train ex. (120)                                | 100.00%      | 98.66%         | 98.67%   | 8.12 min      | A100    |
| 10   | gpt2-small (124M)  | random     | last                     | all              | longest train ex. (120)                                | 100.00%      | 96.64%         | 93.67%   | 0.69 min      | A100    |
| 11   | gpt2-small (124M)  | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 97.32%         | 96.67%   | 0.75 min      | A100    |
| 12   | gpt2-xl (1558M)    | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 98.66%         | 98.33%   | 5.79 min      | A100    |
| 13   | gpt2-small (124M)  | pretrained | last                     | last_block       | context length (1024)                                  | 83.08%       | 87.92%         | 78.33%   | 2.46 min      | A100    |
| 14   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 1)                    | 100.00%      | 98.66%         | 98.00%   | 1.75 min      | A100    |
| 15   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 8)                    | 99.33%       | 98.66%         | 98.33%   | 1.70 min      | A100    |
| 16   | gpt2-small (124M)  | pretrained | last                     | last_block       | flexible (last non-padding position)                   | 99.42%       | 98.66%         | 98.33%   | 0.30 min      | A100    |
| 17   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120); but no causal mask            | 99.23%       | 98.66%         | 95.33%   | 0.29 min      | A100    |
| 18   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120) and `ignore_index` for padding | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 19   | gpt2-small (124M)  | pretrained | last + pooled embeddings | last_block       | longest train ex. (120)                                | 97.79%       | 99.33%         | 96.33%   | 0.32 min      | A100    |

&nbsp;

### 用法

可用以下代码复现实验：

- Row 1: `python additional_experiments.py`
- Row 2: `python additional_experiments.py --trainable_token_pos first`
- Row 3: `python additional_experiments.py --trainable_layers last_layer`
- Row 4: `python additional_experiments.py --trainable_layers last_two_blocks`
- Row 5: `python additional_experiments.py --trainable_layers all`
- Row 6: `python additional_experiments.py --model_size "gpt2-medium (355M)"`
- Row 7: `python additional_experiments.py --model_size "gpt2-large (774M)"`
- Row 8: `python additional_experiments.py --model_size "gpt2-xl (1558M)"`
- Row 9: `python additional_experiments.py --model_size "gpt2-xl (1558M)"--trainable_layers all`
- Row 10: `python additional_experiments.py --weights random --trainable_layers all`
- Row 11: `python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- Row 12: `python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 8 --model_size "gpt2-xl (1558M)"`
- Row 13: `python additional_experiments.py --context_length "model_context_length"`
- Row 14: `python additional_experiments.py --no_padding --batch_size 1`
- Row 15: `python additional_experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- Row 16: `python additional_experiments.py --trainable_token_pos "flexible"`
- Row 17: `python additional_experiments.py --disable_causal_mask`
- Row 18: `python additional_experiments.py --ignore_index 50256`
- Row 19: `python additional_experiments.py --average_embeddings`

我故意保持 LLM 与数据集较小，以便在无 GPU 时可在普通笔记本（如 MacBook Air M3）上约 15 分钟（默认设置）完成训练。

&nbsp;

### 解读

1. **训练最后与第一个输出 token 位置（第 1 行 vs. 第 2 行）**：训练最后输出 token 位置相比第一个 token 性能明显更好。由于因果自注意力掩码，这一提升在预期之中。
2. **训练最后一个 Transformer 块 vs. 最后一层（第 1 行 vs. 第 3 行）**：训练整个最后一个 Transformer 块也明显优于仅训练最后一层。
3. **训练最后一个 vs. 最后两个 Transformer 块（第 1 行 vs. 第 4 行）**：训练最后两个 Transformer 块而非仅最后一个块，准确率提升约 3.33%。
4. **训练最后一个 Transformer 块 vs. 全部层（第 1 行 vs. 第 5 行）**：训练全部层相比仅训练最后一个块约有 ~2% 提升，但训练时长几乎增至三倍；且不如仅训练 12 个块中最后两个。
5. **使用更大预训练模型（第 1 行 vs. 6，以及第 1 行 vs. 7、8）**：使用约 3 倍大的预训练模型结果更差；约 5 倍大的模型则如预期优于初始模型；约 12 倍大的模型进一步提升预测性能。（medium 模型可能预训练不足，或特定微调配置对该模型效果不佳。）
6. **随机权重 vs. 预训练权重（第 1、5 行 vs. 第 10 行）**：随机权重模型仅略差于预训练权重（约 3% 与 1.3%）。
7. **使用 LoRA（低秩适配）vs. 训练全部层（第 11 行 vs. 5，第 12 行 vs. 9）**：冻结模型并添加可训练 LoRA 层（见 [附录 E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb)）是训练全部参数的可行替代，甚至提升约 1 个百分点（第 11 行 vs. 5）。LoRA 时训练与验证准确率差距约低 1%，可能因过拟合更少；且更新参数更少、更省内存。训练更大模型时（第 12 行 vs. 9），LoRA 也更快（5.79 分钟 vs. 8.12 分钟）。
8. **填充至完整上下文长度 vs. 最长训练样本（第 1 行 vs. 第 13 行）**：将输入填充至模型支持的完整上下文长度明显更差。
9. **填充 vs. 不填充（第 1 行 vs. 14、15 与 16）**：`--no_padding` 关闭数据集填充，因输入长度可变需 batch size 为 1 训练；测试准确率更好但训练更久。第 15 行额外启用 8 步梯度累积以匹配其他实验的有效 batch size，有助于减轻过拟合并略升测试准确率。第 16 行仍填充，但 token 位置取最后非填充 token。第 16 行在数学上应接近使用梯度累积的第 15 行；但因 token 数不等时梯度累积存在难点，可能有小幅差异（见[此文](https://unsloth.ai/blog/gradient)）。
10. **禁用因果注意力掩码（第 1 行 vs. 17）**：禁用多头注意力中的因果掩码，使所有 token 可互相关注；相比带因果掩码的 GPT 模型准确率略升。
11. **在损失与反传中忽略填充索引（第 1 行 vs. 18）**：`--ignore_index 50256` 在 PyTorch 的 `cross_entropy` 中排除 `<|endoftext|>` 填充 token。此处因已替换输出层，二分类 token ID 仅为 0 或 1，故无影响；但该设置在第 7 章指令微调时有用。
12. **对所有 token 嵌入取平均（第 1 行 vs. 19）**：`--average_embeddings` 对所有 token 嵌入取平均。默认情况下仅使用 `--trainable_token_pos` 指定位置的输出嵌入（例如最后 token）。启用后将对所有 token 嵌入做均值池化到 `--trainable_token_pos` 所选位置（默认最后 token）。可见性能从 95.00% 升至 96.33%，运行时间仅略增（0.28 至 0.32 分钟），实践中值得考虑。
