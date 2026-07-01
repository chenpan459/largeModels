# 11 - 与 llama.cpp / kefu-kb 对照

## 推理双栈

| 维度 | vLLM | llama.cpp |
|------|------|-----------|
| 权重 | HF / Safetensors | GGUF |
| 强项 | 高并发 PagedAttention | 轻量、多量化 |
| API | `vllm serve` | `llama-server` |

## 概念映射

| vLLM | llama.cppDoc |
|------|--------------|
| BlockPool / PagedAttention | 16-kv-cache-memory |
| Continuous Batching | 17-batch-system |
| OpenAI API | 12-server |

## kefu-kb

当前用 llama-server；可改 `config.yaml` 的 `base_url` 指向 vLLM（需 HF 权重）。

## LlamaIndex

`OpenAILike(api_base="http://127.0.0.1:8000/v1")` 兼容两者。

## Megatron

训练 → Megatron Bridge → HF → vLLM serve。
