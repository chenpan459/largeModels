# 14 - ggml-opt 与线程系统

## 1. 概述

GGML 除推理外还支持 **小规模训练**，核心模块：

| 文件 | 职责 |
|------|------|
| `src/ggml-opt.cpp` | 优化器、损失、训练图构建 |
| `src/ggml-threading.cpp` | 线程池（CPU Backend 使用） |
| `ggml.c` | `ggml_build_backward_expand`、梯度相关 |

llama.cpp **推理不使用** backward；examples（gpt-2、mnist）和 research 使用 opt 路径。

## 2. 训练相关算子

| 算子 | 用途 |
|------|------|
| `GGML_OP_OPT_STEP_ADAMW` | AdamW 优化步 |
| `GGML_OP_OPT_STEP_SGD` | SGD 优化步 |
| `GGML_OP_CROSS_ENTROPY_LOSS` | 交叉熵损失 |
| `GGML_OP_CROSS_ENTROPY_LOSS_BACK` | 损失反向 |

CUDA 实现：`ggml-cuda/opt-step*.cu`

## 3. 训练数据流

```
1. ggml_init() + 创建 PARAM flag 权重 tensor
2. forward: ggml_build_forward_expand(gf, loss_node)
3. backward: ggml_build_backward_expand(gf, loss_node)
4. optimizer: ggml_opt_step_adamw / sgd 节点加入图
5. ggml_graph_compute 或 backend_sched 执行
6. 重复 epoch
```

`ggml_cgraph` 训练字段：

- `grads[]`：每节点梯度 tensor
- `grad_accs[]`：梯度累加器
- `GGML_TENSOR_FLAG_PARAM`：可训练参数
- `GGML_TENSOR_FLAG_LOSS`：损失节点

## 4. ggml-opt API（概念）

`ggml-opt.cpp` 提供：

- 优化器状态 tensor 管理（momentum、variance）
- 学习率、weight decay 参数（编码在 `op_params`）
- 与 `ggml_backend_sched` 兼容的多设备训练（实验性）

具体 API 见 `include/ggml.h` 中 `ggml_opt_*` 声明及 `examples/gpt-2/`。

## 5. 线程池（ggml-threading.cpp）

### 结构

```c
struct ggml_threadpool;  // 不透明

ggml_threadpool_t ggml_threadpool_new(params);
void ggml_threadpool_free(ggml_threadpool_t);
void ggml_threadpool_pause/resume(...);
```

### 与 CPU Backend 集成

`ggml_graph_plan()` 设置 `cplan.threadpool`：

- 替代或补充 OpenMP
- 支持 **affinity / cpumask**（绑定物理核）
- disposable threadpool：短任务专用

llama.cpp `-t N` → `n_threads` → graph plan。

### OpenMP vs Threadpool

| | OpenMP | Threadpool |
|---|--------|------------|
| CMake | `GGML_OPENMP=ON` | 始终可用 |
| 粒度 | 节点内 parallel | graph plan 控制 |
| Affinity | 依赖 OMP 环境变量 | ggml cpumask API |

## 6. NUMA

`ggml_numa_init()`（可选编译）：

- 多 socket 服务器绑定内存与线程
- 大模型 CPU 推理场景

## 7. 与推理路径对比

| | 推理（llama） | 训练（examples） |
|---|---------------|------------------|
| 建图 | forward only | forward + backward |
| 执行 | sched_graph_compute_async | graph_compute / sched |
| 内存 | gallocr in-place | 需存储 grads |
| Backend | GPU 优先 | 通常 CPU 或单 GPU |

## 8. 示例入口

| 示例 | 路径 | 说明 |
|------|------|------|
| simple | `examples/simple/` | 最小张量 API |
| mnist | `examples/mnist/` | 分类训练 |
| gpt-2 | `examples/gpt-2/` | 小 GPT 训练 |
| magika | `examples/magika/` | 文件类型检测 |

## 相关文档

- [03-tensor-graph.md](./03-tensor-graph.md)
- [08-backend-cpu.md](./08-backend-cpu.md)
- [02-architecture.md](./02-architecture.md)
