# 08 - 并行与分布式推理

## ParallelConfig

`config.py` → `ParallelConfig`：

| 字段 | 含义 |
|------|------|
| `tensor_parallel_size` (TP) | 张量并行度 |
| `pipeline_parallel_size` (PP) | 流水线并行度 |
| `data_parallel_size` (DP) | 数据并行副本数 |
| `world_size` | 总 worker 数 = TP × PP × DP |
| `distributed_executor_backend` | `mp`（multiproc）或 `ray` |

```python
world_size = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
```

## Tensor Parallelism（TP）

### 原理

与 Megatron-LM 相同数学：

| 层 | 分片方式 | 通信 |
|----|----------|------|
| ColumnParallelLinear | 按 output dim 切分 | 无（或 all-gather output） |
| RowParallelLinear | 按 input dim 切分 | all-reduce output |
| VocabParallelEmbedding | 按 vocab 切分 | all-reduce 或 gather |

实现：`model_executor/layers/linear.py`

```python
class ColumnParallelLinear:
    # weight: [output_size/tp, input_size]
    # forward: local matmul

class RowParallelLinear:
    # weight: [output_size, input_size/tp]
    # forward: local matmul → all_reduce
```

### 分布式状态

`distributed/parallel_state.py`：

```python
initialize_model_parallel(
    tensor_model_parallel_size=TP,
    pipeline_model_parallel_size=PP,
)
# 创建 TP/PP/DP process group
```

Collective 原语：

- `tensor_model_parallel_all_reduce`
- `tensor_model_parallel_all_gather`
- `gather_from_tensor_model_parallel_region`

### V1 MultiprocExecutor

`v1/executor/multiproc_executor.py`：

```
主进程 Executor
  ├─ Worker 0 (rank 0, GPU 0) → GPUModelRunner
  ├─ Worker 1 (rank 1, GPU 1) → GPUModelRunner
  └─ ...
world_size == tensor_parallel_size  # 单机 TP
```

- 每 GPU 一个子进程
- `rpc_broadcast_mq` 广播 SchedulerOutput
- Forward 内 TP collective（NCCL）
- 仅 rank 0 返回 ModelRunnerOutput

启动：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2
# 或 -tp 2
```

Worker 类自动选择（`platforms/cuda.py:118-142`）：

```python
VLLM_USE_V1=True → vllm.v1.worker.gpu_worker.Worker
VLLM_USE_V1=False → vllm.worker.worker.Worker
```

## Pipeline Parallelism（PP）

### V0

较完整支持：

- `pipeline_model_parallel` process group
- Stage 间 P2P 传 `IntermediateTensors`
- `EngineCore.step_with_batch_queue()` 消除 bubble

### V1

- **必须** `--distributed-executor-backend ray` 且 `pipeline_parallel_size > 1`
- Oracle 检查（`arg_utils._is_v1_supported_oracle`）：非 Ray backend + PP → 拒绝 V1
- `GPUModelRunner` 中 PP rank 分支：
  ```python
  if not get_pp_group().is_first_rank:
      # 接收 intermediate_tensors
  if not get_pp_group().is_last_rank:
      return IntermediateTensors(...)  # 传给下一 stage
  ```

```bash
# V1 PP 示例（需 Ray 集群）
vllm serve model --pipeline-parallel-size 2 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray
```

## Data Parallelism（DP）

V1 部分支持：

- `data_parallel_size > 1` → `DPAsyncMPClient`（多 EngineCore 实例）
- 每 DP rank 独立 Scheduler + Worker 组
- 负载不均时 `execute_dummy_batch()` 保持 collective 同步
- `LLMEngine` init 时建立 DP group（`llm_engine.py:66-69`）

部署层通常用多 replica + 负载均衡代替 in-process DP。

## Expert Parallelism（EP）

MoE 模型（Mixtral、DeepSeek 等）：

- `FusedMoE` + `device_communicators/all2all.py`
- Token dispatch：router 决定 expert → all-to-all 通信
- 可与 TP 组合

## Ray 多节点

`v1/executor/ray_distributed_executor.py`：

```
Ray Cluster
  ├─ Ray Actor: Worker rank 0 (node A, GPU 0-3)
  ├─ Ray Actor: Worker rank 1 (node A, GPU 4-7)
  ├─ Ray Actor: Worker rank 2 (node B, GPU 0-3)
  └─ ...
```

- 跨节点 NCCL（需正确 `VLLM_HOST_IP`、NCCL 配置）
- 适合大模型多机 TP 或 TP+PP
- 启动前需 `ray start --head` / join cluster

## 分布式初始化流程

每个 Worker 子进程：

```python
distributed_init_method = get_distributed_init_method("127.0.0.1", port)
torch.distributed.init_process_group(
    backend="nccl", rank=rank, world_size=world_size, ...)
initialize_model_parallel(tensor_model_parallel_size=TP, ...)
```

主进程 Executor 协调 RPC；Worker 自行 init rank。

## 通信与 NCCL 调优

| 环境变量 | 作用 |
|----------|------|
| `CUDA_VISIBLE_DEVICES` | 可见 GPU |
| `NCCL_DEBUG` | 通信 debug 日志 |
| `NCCL_IB_DISABLE` | 禁用 IB（调试） |
| `VLLM_HOST_IP` | 多机 IP 绑定 |
| `VLLM_SKIP_P2P_CHECK` | 跳过 P2P 检查 |

## 与 Megatron 训练并行对照

| 训练（Megatron） | 推理（vLLM） |
|----------------|--------------|
| TP + PP + DP + EP | 主要 TP（+ evolving PP/DP） |
| 3D 并行 | 推理侧重 TP 降单 request 延迟 |
| 权重分片训练 | 同分片加载推理 |
| Checkpoint | Megatron Bridge → HF → vLLM |

Megatron 训练 checkpoint 需经 **Megatron Bridge** 转为 HuggingFace 格式后 vLLM 加载。

## 多 API 实例部署

| 模式 | 说明 |
|------|------|
| 单进程多 GPU | TP 一体，一个 `vllm serve` |
| 多 replica | 多个独立进程 + nginx/Envoy LB |
| K8s | 每 pod 一个 vLLM instance，HPA 扩缩 |

vLLM 基础版无内置 request 级 LB；DP 主要在 engine 内部。

## 多模态 + TP 注意

- Vision tower 可能与 LLM 不同 TP 策略
- Encoder cache 不跨 TP rank 共享
- 大图片 + 高 TP 需注意 encoder 内存

## 当前限制（源码为准）

| 限制 | 说明 |
|------|------|
| V1 单机 executor | `world_size == TP`（无 PP） |
| V1 PP | 仅 Ray backend |
| V1 DP | 实验性，dummy batch 开销 |
| EP + TP | 模型相关，查具体架构支持 |

跟踪 upstream CHANGELOG 获取最新 PP/DP 进展。

## 快速命令

```bash
# 2 卡 TP
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-7B-Instruct -tp 2

# 4 卡 TP + 高并发
vllm serve model -tp 4 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.95

# 指定 NCCL
NCCL_DEBUG=INFO vllm serve model -tp 8
```

## 关键源码

| 主题 | 文件 |
|------|------|
| TP linear | `model_executor/layers/linear.py` |
| Parallel state | `distributed/parallel_state.py` |
| Multiproc executor | `v1/executor/multiproc_executor.py` |
| Ray executor | `v1/executor/ray_distributed_executor.py` |
| PP in runner | `v1/worker/gpu_model_runner.py` |
| MoE all2all | `distributed/device_communicators/` |
