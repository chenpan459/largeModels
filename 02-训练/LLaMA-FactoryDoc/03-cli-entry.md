# 03 — CLI 与入口

## CLI 命令

安装后注册两个命令别名：

```bash
llamafactory-cli   # 完整命令
lmf                # 快捷别名
```

入口：`src/llamafactory/cli.py:main()` → `launcher.py:launch()`

## 子命令一览

| 命令 | 处理器 | 说明 |
|------|--------|------|
| `train` | `train.tuner.run_exp()` | 训练模型 |
| `export` | `train.tuner.export_model()` | 合并 LoRA 并导出 |
| `api` | `api.app.run_api()` | 启动 OpenAI 兼容 API |
| `chat` | `chat.chat_model.run_chat()` | CLI 对话 |
| `webui` | `webui.interface.run_web_ui()` | 启动 LLaMA Board |
| `webchat` | `webui.interface.run_web_demo()` | 仅聊天 Web UI |
| `env` | `extras.env.print_env()` | 打印环境信息 |
| `version` | — | 打印版本 |
| `help` | — | 打印帮助 |

## 常用命令示例

```bash
# LoRA 微调
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# CLI 覆盖 YAML 参数
llamafactory-cli train config.yaml learning_rate=1e-5 logging_steps=1

# 纯 CLI 参数（无 YAML）
llamafactory-cli train --model_name_or_path Qwen/Qwen3-4B --stage sft ...

# 启动 Web UI
llamafactory-cli webui

# 启动 API 服务
llamafactory-cli api

# CLI 对话
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml

# 合并 LoRA 导出
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

## 入口脚本

除 CLI 外，项目提供三个薄封装脚本，可直接用 Python 调用：

| 文件 | 调用 | 说明 |
|------|------|------|
| `src/train.py` | `run_exp()` | 等同 `llamafactory-cli train` |
| `src/api.py` | 创建 ChatModel + Uvicorn | 等同 `llamafactory-cli api` |
| `src/webui.py` | 启动 Gradio | 等同 `llamafactory-cli webui` |

生产环境推荐使用 `llamafactory-cli`，因为它包含分布式启动逻辑。

## 分布式训练自动启动

`launcher.py` 在检测到多 GPU 时会自动用 `torchrun` 包装 `train` 命令：

```python
# launcher.py 核心逻辑（简化）
if command == "train" and (
    is_env_enabled("FORCE_TORCHRUN")
    or (get_device_count() > 1 and not use_ray() and not use_kt())
):
    subprocess.run(["torchrun", "--nproc_per_node", N, __file__, "train", ...])
```

### 分布式环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NNODES` | `1` | 节点数 |
| `NODE_RANK` | `0` | 当前节点 rank |
| `NPROC_PER_NODE` | GPU 数量 | 每节点进程数 |
| `MASTER_ADDR` | `127.0.0.1` | 主节点地址 |
| `MASTER_PORT` | 自动分配 | 主节点端口 |
| `RDZV_ID` | — | 弹性训练 rendezvous ID |
| `MAX_RESTARTS` | `0` | 故障恢复最大重启次数 |
| `FORCE_TORCHRUN` | — | 强制使用 torchrun（单卡也可用） |

### 多节点示例

```bash
# 节点 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 NPROC_PER_NODE=8 \
  llamafactory-cli train config.yaml

# 节点 1
NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 NPROC_PER_NODE=8 \
  llamafactory-cli train config.yaml
```

## 特殊模式环境变量

| 变量 | 效果 |
|------|------|
| `USE_V1=1` | 切换到 v1 实验架构 |
| `USE_MCA=1` | 启用 Megatron-core 适配器，强制 torchrun |
| `OPTIM_TORCH=1` | 优化 DDP 内存（expandable_segments 等） |

## launcher.py 命令分发流程

```
sys.argv[1] → command
    │
    ├── train + 多 GPU → torchrun 包装 → run_exp()
    ├── train + 单 GPU → run_exp()
    ├── export → export_model()
    ├── api → run_api()
    ├── chat → run_chat()
    ├── webui → run_web_ui()
    ├── webchat → run_web_demo()
    ├── env → print_env()
    └── version / help
```

## train 子命令内部流程

```
run_exp(args)
  ├── read_args()              # 读 YAML/JSON/CLI
  ├── get_ray_args()           # Ray 集群（可选）
  └── _training_function()
        ├── get_train_args()   # 解析 + 校验
        ├── 注册 callbacks
        └── 按 stage 路由:
              pt  → run_pt()
              sft → run_sft()
              rm  → run_rm()
              ppo → run_ppo()
              dpo → run_dpo()
              kto → run_kto()
```

特殊后端路由（优先级高于默认 stage）：

- `use_hyper_parallel=True` → `hyper_parallel/run_{pt,sft}.py`
- `use_mca=True` → `mca/run_{pt,sft,dpo}.py`
