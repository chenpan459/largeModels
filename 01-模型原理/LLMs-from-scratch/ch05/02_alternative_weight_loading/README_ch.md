# 加载预训练权重的替代方案

本文件夹提供在 OpenAI 权重不可用时使用的替代加载策略。

- [weight-loading-pytorch_ch.ipynb](weight-loading-pytorch_ch.ipynb)：（推荐）从作者由原始 TensorFlow 权重转换得到的 PyTorch state dict 加载权重

- [weight-loading-hf-transformers_ch.ipynb](weight-loading-hf-transformers_ch.ipynb)：通过 `transformers` 库从 Hugging Face Model Hub 加载权重

- [weight-loading-hf-safetensors_ch.ipynb](weight-loading-hf-safetensors_ch.ipynb)：直接通过 `safetensors` 库从 Hugging Face Model Hub 加载权重（无需实例化 Hugging Face Transformer 模型）
