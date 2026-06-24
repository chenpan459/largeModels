# 04 - 模型架构层 (src/models/)

## 1. 模块概述

`src/models/` 目录包含 **134 个模型架构实现**，每个文件对应一种或一组 closely related 的模型架构。这是 llama.cpp 支持 100+ 模型的核心扩展机制。

## 2. 设计模式

### 2.1 一架构一文件

每个模型类继承自 `llama_model` 基类，实现三个核心方法：

```cpp
class llama_model_xxx : public llama_model {
    // 1. 从 GGUF metadata 加载超参数
    void load_arch_hparams(llama_model_loader & ml);

    // 2. 声明并创建权重张量
    void load_arch_tensors(llama_model_loader & ml);

    // 3. 构建前向计算图
    ggml_cgraph * build_graph(llm_graph_result & result);
};
```

### 2.2 架构注册

`src/llama-arch.h` 定义 `enum llm_arch`（140+ 枚举值），`src/llama-arch.cpp` 负责：

- GGUF metadata 中 `general.architecture` 字段 -> `llm_arch` 映射
- 各架构的 KV 键名映射 (`enum llm_kv`)
- 张量命名规范 (`enum llm_tensor`)

### 2.3 Graph 构建基类

`src/models/models.h` 提供共享基类：

| 基类 | 用途 |
|------|------|
| `llm_graph_context` | 通用 graph 构建上下文 |
| `llm_build_mamba_base` | Mamba/Mamba2 层构建 |
| `llm_build_delta_net_base` | Delta Net (Gated DeltaNet) 层构建 |
| `llm_build_rwkv_base` | RWKV 层构建 |

## 3. 模型文件分类

### 3.1 Transformer 系列

| 文件 | 架构 | 代表模型 |
|------|------|----------|
| `llama.cpp` | LLM_ARCH_LLAMA | LLaMA 1/2/3, Mistral, Vicuna |
| `llama4.cpp` | LLM_ARCH_LLAMA4 | LLaMA 4 |
| `qwen.cpp` | LLM_ARCH_QWEN | Qwen 1.x |
| `qwen2.cpp` | LLM_ARCH_QWEN2 | Qwen 2 |
| `qwen3.cpp` | LLM_ARCH_QWEN3 | Qwen 3 |
| `qwen35.cpp` | LLM_ARCH_QWEN35 | Qwen 3.5 |
| `gemma.cpp` | LLM_ARCH_GEMMA | Gemma 1 |
| `gemma2.cpp` | LLM_ARCH_GEMMA2 | Gemma 2 |
| `gemma3.cpp` | LLM_ARCH_GEMMA3 | Gemma 3 |
| `gemma4.cpp` | LLM_ARCH_GEMMA4 | Gemma 4 |
| `deepseek.cpp` | LLM_ARCH_DEEPSEEK | DeepSeek V1 |
| `deepseek2.cpp` | LLM_ARCH_DEEPSEEK2 | DeepSeek V2/V3 (MLA) |
| `deepseek32.cpp` | LLM_ARCH_DEEPSEEK32 | DeepSeek V3.2 |
| `phi2.cpp` / `phi3.cpp` | LLM_ARCH_PHI2/3 | Phi-2/3/3.5 |
| `glm4.cpp` | LLM_ARCH_GLM4 | GLM-4 |
| `internlm2.cpp` | LLM_ARCH_INTERNLM2 | InternLM2 |
| `falcon.cpp` | LLM_ARCH_FALCON | Falcon |
| `gpt2.cpp` | LLM_ARCH_GPT2 | GPT-2 |
| `gptneox.cpp` | LLM_ARCH_GPTNEOX | GPT-NeoX/Pythia |
| `olmo.cpp` / `olmo2.cpp` | LLM_ARCH_OLMO/2 | OLMo |
| `stablelm.cpp` | LLM_ARCH_STABLELM | StableLM |
| `baichuan.cpp` | LLM_ARCH_BAICHUAN | Baichuan |
| `xverse.cpp` | LLM_ARCH_XVERSE | Xverse |
| `orion.cpp` | LLM_ARCH_ORION | Orion |
| `bloom.cpp` | LLM_ARCH_BLOOM | BLOOM |
| `mpt.cpp` | LLM_ARCH_MPT | MPT |
| `starcoder.cpp` / `starcoder2.cpp` | STARCODER | StarCoder |
| `granite.cpp` | LLM_ARCH_GRANITE | IBM Granite |
| `openelm.cpp` | LLM_ARCH_OPENELM | OpenELM |
| `chatglm.cpp` | LLM_ARCH_CHATGLM | ChatGLM |
| `exaone.cpp` / `exaone4.cpp` | EXAONE | EXAONE |
| `plamo.cpp` / `plamo2.cpp` / `plamo3.cpp` | PLAMO | PLaMo |
| `minicpm.cpp` / `minicpm3.cpp` | MINICPM | MiniCPM |
| `bitnet.cpp` | LLM_ARCH_BITNET | BitNet b1.58 |
| `lfm2.cpp` | LLM_ARCH_LFM2 | Liquid LFM2 |

### 3.2 MoE 系列

| 文件 | 架构 | 代表模型 |
|------|------|----------|
| `qwen2moe.cpp` | QWEN2MOE | Qwen2-MoE |
| `qwen3moe.cpp` | QWEN3MOE | Qwen3-MoE |
| `qwen35moe.cpp` | QWEN35MOE | Qwen3.5-MoE |
| `glm4-moe.cpp` | GLM4_MOE | GLM-4-MoE |
| `granite-moe.cpp` | GRANITE_MOE | Granite MoE |
| `dbrx.cpp` | DBRX | DBRX |
| `arctic.cpp` | ARCTIC | Snowflake Arctic |
| `olmoe.cpp` | OLMOE | OLMoE |
| `phimoe.cpp` | PHIMOE | PhiMoE |
| `bailingmoe.cpp` / `bailingmoe2.cpp` | BAILINGMOE | BailingMoe |
| `hunyuan-moe.cpp` | HUNYUAN_MOE | Hunyuan MoE |
| `openai-moe.cpp` | OPENAI_MOE | gpt-oss |
| `afmoe.cpp` | AFMOE | AFMoE |
| `grovemoe.cpp` | GROVEMOE | GroveMoE |
| `cohere2moe.cpp` | COHERE2MOE | Cohere Command-R MoE |

### 3.3 状态空间模型

| 文件 | 架构 | 代表模型 |
|------|------|----------|
| `mamba.cpp` | MAMBA | Mamba |
| `mamba2.cpp` | MAMBA2 | Mamba-2 |
| `mamba-base.cpp` | - | Mamba 共享基类 |
| `rwkv6.cpp` / `rwkv7.cpp` | RWKV6/7 | RWKV-6/7 |
| `rwkv6-base.cpp` / `rwkv7-base.cpp` | - | RWKV 共享基类 |
| `jamba.cpp` | JAMBA | Jamba (Attn+SSM) |
| `falcon-h1.cpp` | FALCON_H1 | Falcon-H1 |
| `granite-hybrid.cpp` | GRANITE_HYBRID | Granite Hybrid |
| `delta-net-base.cpp` | - | Gated DeltaNet 基类 |
| `kimi-linear.cpp` | KIMI_LINEAR | Kimi Linear |

### 3.4 多模态

| 文件 | 架构 | 代表模型 |
|------|------|----------|
| `qwen2vl.cpp` | QWEN2VL | Qwen2-VL |
| `qwen3vl.cpp` / `qwen3vlmoe.cpp` | QWEN3VL | Qwen3-VL |
| `gemma3n.cpp` | GEMMA3N | Gemma 3n (多模态) |
| `cogvlm.cpp` | COGVLM | CogVLM |
| `chameleon.cpp` | CHAMELEON | Chameleon |
| `hunyuan-vl.cpp` | HUNYUAN_VL | Hunyuan-VL |
| `mistral3.cpp` / `mistral4.cpp` | MISTRAL3/4 | Mistral 3/4 Vision |
| `paddleocr.cpp` | PADDLEOCR | PaddleOCR |
| `deepseek2ocr.cpp` | DEEPSEEK2OCR | DeepSeek OCR |

### 3.5 Embedding / Rerank

| 文件 | 架构 | 代表模型 |
|------|------|----------|
| `bert.cpp` | BERT | BERT |
| `modern-bert.cpp` | MODERN_BERT | ModernBERT |
| `nomic-bert.cpp` | NOMIC_BERT | Nomic Embed |
| `jina-bert-v2.cpp` / `jina-bert-v3.cpp` | JINA_BERT | Jina Embeddings |
| `gemma-embedding.cpp` | GEMMA_EMBEDDING | Gemma Embedding |
| `llama-embed.cpp` | LLAMA_EMBED | LLaMA Embedding |
| `pangu-embed.cpp` | PANGU_EMBED | Pangu Embed |
| `t5encoder.cpp` | T5ENCODER | T5 Encoder |

### 3.6 特殊架构

| 文件 | 架构 | 代表模型 |
|------|------|----------|
| `llada.cpp` / `llada-moe.cpp` | LLADA | LLaDA (Diffusion LM) |
| `dream.cpp` | DREAM | Dream (Diffusion) |
| `t5.cpp` | T5 | T5 (Encoder-Decoder) |
| `eagle3.cpp` | EAGLE3 | EAGLE3 (Speculative) |
| `wavtokenizer-dec.cpp` | WAVTOKENIZER | WavTokenizer |
| `talkie.cpp` | TALKIE | Talkie (TTS) |
| `maincoder.cpp` | MAINCODER | MainCoder (FIM) |
| `refact.cpp` | REFACT | Refact (Code) |
| `codeshell.cpp` | CODESHELL | CodeShell |
| `dots1.cpp` | DOTS1 | Dots1 |
| `rnd1.cpp` | RND1 | RND1 |
| `mellum.cpp` | MELLUM | Mellum (JetBrains) |
| `seed-oss.cpp` | SEED_OSS | Seed-OSS |
| `smollm3.cpp` | SMOLLM3 | SmolLM3 |
| `smallthinker.cpp` | SMALLTHINKER | SmallThinker |
| `nemotron.cpp` / `nemotron-h.cpp` | NEMOTRON | Nemotron |
| `step35.cpp` | STEP35 | Step3.5 |
| `mimo2.cpp` | MIMO2 | Mimo2 |
| `minimax-m2.cpp` | MINIMAX_M2 | MiniMax M2 |
| `ernie4-5.cpp` / `ernie4-5-moe.cpp` | ERNIE4_5 | ERNIE 4.5 |
| `grok.cpp` | GROK | Grok-1 |
| `deci.cpp` | DECI | DeciLM |
| `command-r.cpp` / `cohere2.cpp` | COMMAND_R | Command-R |
| `jais.cpp` / `jais2.cpp` | JAIS | Jais |
| `plm.cpp` | PLM | PLM |
| `apertus.cpp` | APERTUS | Apertus |
| `glm-dsa.cpp` | GLM_DSA | GLM DSA |
| `arcee.cpp` | ARCEE | Arcee |
| `hunyuan-dense.cpp` | HUNYUAN_DENSE | Hunyuan Dense |
| `lfm2moe.cpp` | LFM2MOE | LFM2 MoE |
| `eurobert.cpp` | - | EuroBERT |
| `neo-bert.cpp` | NEO_BERT | NeoBERT |
| `nomic-bert-moe.cpp` | NOMIC_BERT_MOE | Nomic MoE |
| `nemotron-h-moe.cpp` | NEMOTRON_H_MOE | Nemotron-H MoE |
| `exaone-moe.cpp` | EXAONE_MOE | EXAONE MoE |
| `rwkv6qwen2.cpp` | RWKV6QWEN2 | QRWKV6 |
| `arwkv7.cpp` | ARWKV7 | ARWKV7 |

## 4. 添加新模型的步骤

1. 在 `llama-arch.h` 添加 `LLM_ARCH_XXX` 枚举
2. 在 `llama-arch.cpp` 注册架构名和 KV 键
3. 创建 `src/models/xxx.cpp` 实现三个核心方法
4. 在 `src/models/models.h` 声明模型类
5. 在 `llama-model.cpp` 的工厂函数中注册
6. 添加 `conversion/xxx.py` 转换脚本 (HF -> GGUF)
7. 更新 `convert_hf_to_gguf.py` 的模型映射

详细指南: `docs/development/HOWTO-add-model.md`

## 5. 典型 Transformer 层结构 (以 LLaMA 为例)

```
Input Tokens
    |
    v
[Token Embedding]
    |
    v
For each layer i:
    [RMS Norm] -> [Self-Attention (Q,K,V,RoPE)] -> [Residual]
    [RMS Norm] -> [FFN (Gate*Silu(Up)) -> Down] -> [Residual]
    |
    v
[Output RMS Norm]
    |
    v
[Output Projection] -> Logits
```

MoE 变体将 FFN 替换为 Expert Gate + Multiple Experts。

## 6. 文件规模

大部分模型文件在 100-500 行之间，复杂架构 (DeepSeek2 MLA, Qwen3-VL) 可达 1000+ 行。最大的通用模块是 `models.h` (~1900 行)，包含所有基类和共享构建函数。
