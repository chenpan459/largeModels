# 13 - 采样与 Structured Output

## 概述

vLLM 采样发生在 **ModelRunner 末 rank**，在 transformer forward 得到 logits 之后。V0 与 V1 实现路径不同，V1 更 GPU 化且与 structured output 深度集成。

```
Transformer forward
  → compute_logits(sample_hidden_states)   # TP gather 后 logits
  → [optional] apply_grammar_bitmask()    # structured output
  → Sampler / RejectionSampler            # 普通 / spec decode
  → ModelRunnerOutput.sampled_token_ids
  → Scheduler.update_from_output()
```

## V1 采样模块

| 组件 | 路径 |
|------|------|
| `Sampler` | `v1/sample/sampler.py` |
| `SamplingMetadata` | `v1/sample/metadata.py` |
| `TopKTopPSampler` | `v1/sample/ops/topk_topp_sampler.py` |
| Penalties | `v1/sample/ops/penalties.py` |
| Bad words | `v1/sample/ops/bad_words.py` |
| RejectionSampler | `v1/sample/rejection_sampler.py` |
| TPU Sampler | `v1/sample/tpu/sampler.py` |

分发入口：`model_executor/layers/sampler.py` → `get_sampler()`

## V1 采样流水线

`Sampler.forward()`（`sampler.py:23-72`）：

```
1. raw_logprobs = log_softmax(logits)     # 用于 logprobs 输出（惩罚前）
2. logits = float32
3. apply_allowed_token_ids()            # 白名单 mask
4. apply_bad_words()                    # 禁用词
5. apply_logits_bias()                  # token 级 bias
6. apply_penalties()                    # frequency/presence/repetition/min_tokens
7. sample() → greedy 或 TopKTopPSampler
8. gather_logprobs()（若请求 logprobs）
9. 返回 SamplerOutput
```

### 与 V0 的关键差异

| | V0 | V1 |
|---|----|----|
| Logprobs 基准 | 惩罚 **后** 的 logits | 惩罚 **前** 的 raw logits |
| 自定义 logits_processor | `SamplingParams.logits_processors` | **不支持** |
| best_of | 支持 | **不支持** |
| 执行位置 | 部分 CPU | 主要 GPU tensor 操作 |
| Structured output | guided decoding processor | grammar bitmask |

## SamplingParams 字段

| 字段 | 作用 |
|------|------|
| `temperature` | 温度缩放 |
| `top_p` | nucleus sampling |
| `top_k` | top-k 截断 |
| `min_p` | min-p 采样 |
| `presence_penalty` / `frequency_penalty` / `repetition_penalty` | 重复惩罚 |
| `min_tokens` | 最少生成 token（EOS 抑制） |
| `stop` / `stop_token_ids` | 停止条件 |
| `max_tokens` | 最大生成长度 |
| `logprobs` / `prompt_logprobs` | 返回 log 概率 |
| `allowed_token_ids` | 允许 token 白名单 |
| `bad_words` | 禁用词列表 |
| `logits_bias` | token id → bias |
| `n` | 并行采样数（fan-out） |
| `seed` | 随机种子 |
| `guided_decoding` | structured output 参数 |

## TopKTopPSampler

`v1/sample/ops/topk_topp_sampler.py`：

- 默认 PyTorch 实现（`argmax`、`topk`、multinomial）
- 可选 **FlashInfer** 加速（`VLLM_USE_FLASHINFER_SAMPLER=1`）
- FlashInfer ≥ 0.2.3 版本检查
- 返回 int32（FlashInfer）→ 转 int64 → 最终 int32 存储

**注意**：此 FlashInfer 仅用于 **采样**，与 attention FlashInfer 无关。

## Penalties

`v1/sample/ops/penalties.py`：

| 惩罚 | 说明 |
|------|------|
| `frequency_penalty` | 已出现 token 的 logit 减去 freq × penalty |
| `presence_penalty` | 出现过的 token 统一减 penalty |
| `repetition_penalty` | 除法型 repetition（类似 HF） |
| `min_tokens` | 前 N token 禁止 EOS |

输入：`prompt_token_ids` + 已生成 `output_token_ids`（从 InputBatch 获取）。

## Bad Words

`v1/sample/ops/bad_words.py`：

- 将 bad word 最后一个 token 的 logit 设为 `-inf`
- 支持多 token bad word 序列

## LogitsProcessor 层次

vLLM 有两层 "logits 处理"：

### 1. Model 层 LogitsProcessor

`model_executor/layers/logits_processor.py`：

- 在 model forward 内执行
- 主要做 **TP vocab gather**（各 rank partial logits → full logits）
- V0/V1 共用

### 2. 采样层处理

V1：`Sampler` 内的 mask/penalties/bad_words

V0：额外支持 `SamplingParams.logits_processors`（用户自定义 Callable）

V0 OpenAI 路径：`entrypoints/openai/logits_processors.py`

## Structured Output（V1）

路径：`v1/structured_output/`

| 组件 | 文件 |
|------|------|
| `StructuredOutputManager` | `__init__.py` |
| xgrammar backend | `backend_xgrammar.py` |
| guidance backend | `backend_guidance.py` |
| Request 状态 | `request.py` |

### 流程

```
1. Processor 收到 guided_json / guided_regex / guided_choice
2. StructuredOutputManager 编译 grammar → FSM
3. Request 状态 WAITING_FOR_FSM → WAITING（编译完成）
4. Scheduler 输出 grammar_bitmask（每 step 更新）
5. GPUModelRunner.apply_grammar_bitmask(logits, bitmask)
6. Sampler 在 masked logits 上采样
```

与 V0 `guided_decoding` 的区别：

| | V0 | V1 |
|---|----|----|
| 机制 | LogitsProcessor 逐 token mask | Grammar bitmask tensor |
| Backend | lm-format-enforcer 等 | xgrammar / guidance |
| 性能 | CPU processor 开销 | GPU bitmask（更低 overhead） |

配置：`DecodingConfig.guided_decoding_backend = xgrammar | guidance | auto`

## Speculative Decoding 采样

### V1 支持的方法

| 方法 | Proposer | 文件 |
|------|----------|------|
| ngram | `NgramProposer` | `v1/spec_decode/ngram_proposer.py` |
| eagle | `EagleProposer` | `v1/spec_decode/eagle.py` |

配置：`SpeculativeConfig.method`、`num_speculative_tokens`

### RejectionSampler

`v1/sample/rejection_sampler.py`：

```
1. Target model 对 draft tokens 并行算 logits
2. 逐 token 比较 target 与 draft 分布
3. Accept → 继续；Reject → 从 target 重采样，后续 draft 作废
4. 返回 accepted token 列表 + bonus token
```

元数据：`v1/spec_decode/metadata.py` → `SpecDecodeMetadata`

### V0 Spec Decode

`spec_decode/` 目录 — 独立 worker 模式：

- `spec_decode_worker.py` — 包装 target + draft
- Medusa、MLP speculator 等（V1 不支持）

## InputBatch 中的采样状态

`gpu_input_batch.py` 维护 per-request：

```python
temperature, top_p, top_k
generators          # 随机数生成器
greedy_reqs         # temperature ≈ 0
random_reqs         # 需要采样的 request
```

每 step 从 `CachedRequestState` 同步到 GPU tensor。

## 并行采样（n > 1）

`v1/engine/parallel_sampling.py` → `ParentRequest`：

- 一个用户 request fan-out 为多个子 request
- 各自独立 SamplingParams.seed
- OutputProcessor 合并输出

## 调试与常见问题

| 问题 | 排查 |
|------|------|
| 重复生成 | 检查 repetition_penalty |
| 提前 EOS | min_tokens、stop strings |
| logprobs 与 HF 不一致 | V1 用 raw logits（惩罚前） |
| structured output 报错 | grammar 编译失败 → WAITING_FOR_FSM 超时 |
| spec decode 接受率低 | draft 模型不匹配 target |

```bash
# 禁用 FlashInfer sampler 对比
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve model
```

## 关键源码行号

| 主题 | 位置 |
|------|------|
| V1 Sampler | `v1/sample/sampler.py:17-72` |
| TopKTopP | `v1/sample/ops/topk_topp_sampler.py` |
| RejectionSampler | `v1/sample/rejection_sampler.py` |
| Grammar bitmask | `gpu_model_runner.py` apply_grammar_bitmask |
| Processor 限制 | `v1/engine/processor.py:116-121` |
