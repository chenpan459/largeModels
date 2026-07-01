# 08 - 并行与分布式推理

## ParallelConfig

`config.py` — `ParallelConfig`：

| 字段 | 含义 |
|------|------|
| `tensor_parallel_size` | TP 度 |
| `pipeline_parallel_size` | PP 度 |
| `data_parallel_size` | DP（多副本，V1 部分场景） |
| `world_size` | 总 worker 数 |

## Tensor Parallelism（TP）

`distributed/parallel_state.py` — 与 Megatron 类似概念：

- 列并行 Linear、行并行 + allreduce
- `model_executor/layers/linear.py` — `ColumnParallelLinear`、`RowParallelLinear`
- Embedding：`vocab_parallel_embedding.py`

**V1 MultiprocExecutor**：

```python
world_size == tensor_parallel_size  # 单机
```

每 GPU 一进程，collective 在 forward 内完成。

启动：

```bash
vllm serve model --tensor-parallel-size 4
```

## Pipeline Parallelism（PP）

V0 支持较完整；**V1 注释标明 PP not yet implemented in v1**（以当前源码为准）。

PP 路径（V0/evolving）：

- `pipeline_parallel` group
- stage 间 P2P 传 hidden states
- `EngineCore.batch_queue` 为 PP 消除 bubble

## Ray 多节点

`v1/executor/ray_distributed_executor.py`：

- Ray actor 托管 worker
- 跨节点 NCCL
- 适合大模型多机 TP

## 分布式初始化

```python
distributed_init_method = get_distributed_init_method("127.0.0.1", port)
torch.distributed.init_process_group(...)
initialize_model_parallel(tensor_model_parallel_size=...)
```

Worker 子进程各自 init rank。

## 通信原语

- `tensor_model_parallel_all_reduce`
- `gather`、`scatter` 用于 logits / embedding
- MoE：`distributed/device_communicators/` — all-to-all

## 与 Megatron 训练并行对照

| 训练（Megatron） | 推理（vLLM） |
|----------------|--------------|
| TP+PP+DP+EP |  mainly TP（+ evolving PP） |
| 权重分片训练 | 同分片加载推理 |
| 3D 并行 | 推理侧重 TP 减延迟 |

Megatron 训练 checkpoint → **Megatron Bridge** → HF → vLLM。

## 与 beyond-nanogpt

`beyond-nanogpt/mlsys/train_tp.py` 教学 TP；vLLM 在生产 `ColumnParallelLinear` + NCCL 中实现相同数学。

## 多 API 实例

- 单进程多 GPU：TP 一体
- 多 replica：`data_parallel_size` 或多进程 front + LB（部署层）

## 环境变量（节选）

| 变量 | 作用 |
|------|------|
| `CUDA_VISIBLE_DEVICES` | 可见 GPU |
| `NCCL_*` | 通信调优 |
| `VLLM_HOST_IP` | 多机 IP |

## 限制（阅读源码时注意）

- V1 单机 executor 断言 `world_size == TP`
- PP on V1：跟踪 upstream issue/CHANGELOG
- 多模态 + TP：注意 vision tower 分片

## 快速命令

```bash
# 2 卡 TP
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-7B-Instruct -tp 2

# Ray（需集群配置）
# 见官方 docs distributed inference
```
