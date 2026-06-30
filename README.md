# largeModels

大模型学习项目集合，按学习路径分为 6 个模块。

根目录：`/home/cp/work2/largeModels`

## 目录结构

```
largeModels/
├── 01-模型原理/          # Transformer/GPT 从零实现
│   ├── nanoGPT/
│   └── LLMs-from-scratch/
├── 02-训练/              # 预训练、微调、LoRA/DPO
│   ├── how-to-train-your-gpt/
│   └── LLaMA-Factory/
├── 03-推理部署/          # 本地推理、HTTP 服务、高吞吐 serving
│   ├── llama.cpp/        # 含 llama-server (tools/server/)
│   ├── llama.cppDoc/     # llama.cpp 中文源码分析文档
│   └── vllm/
├── 04-量化内核/          # 张量计算、量化 kernel
│   └── ggml/
├── 05-RAG/               # 检索增强生成、知识库
│   └── llama_index/
├── 06-Research/          # 前沿技术、大规模训练
│   ├── beyond-nanogpt/
│   └── Megatron-LM/
├── models/               # 本地模型权重 (GGUF 等)
├── sync-repos.sh         # 一键同步/克隆脚本
└── README.md
```

## 各模块说明

| 模块 | 学什么 | 推荐入口 |
|------|--------|----------|
| **01-模型原理** | Attention、Transformer、GPT 结构 | `nanoGPT/train.py` |
| **02-训练** | 训练循环、微调、LoRA/DPO | `how-to-train-your-gpt/` |
| **03-推理部署** | GGUF 加载、decode、HTTP API | `llama.cppDoc/` + `llama-server` |
| **04-量化内核** | 量化格式、SIMD/CUDA kernel | `ggml/src/ggml-quants.c` |
| **05-RAG** | 向量检索、知识库编排 | `llama_index/docs` |
| **06-Research** | MoE、RL、分布式预训练 | `beyond-nanogpt/` |

## 推荐阅读顺序

```
01-模型原理  →  03-推理部署  →  02-训练  →  04-量化内核  →  05-RAG  →  06-Research
   nanoGPT       llama.cpp       LLaMA-Factory    ggml         llama_index   beyond-nanogpt
```

## 一键同步

```bash
cd /home/cp/work2/largeModels
chmod +x sync-repos.sh
./sync-repos.sh
```

脚本会按分类目录克隆/更新所有仓库，并自动绕过失效的 Git 代理。

## 常用路径

| 用途 | 路径 |
|------|------|
| llama.cpp 编译 | `03-推理部署/llama.cpp/` |
| llama.cpp 文档 | `03-推理部署/llama.cppDoc/` |
| llama-server | `03-推理部署/llama.cpp/build/bin/llama-server` |
| 模型权重 | `models/` |
