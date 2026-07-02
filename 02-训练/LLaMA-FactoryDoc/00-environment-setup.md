# 00 — 环境搭建指南

本文档整合 LLaMA Factory 在 **Ubuntu 22.04** 上的完整环境搭建流程，包含实际部署中遇到的问题与解决方案。

项目路径：`/home/cp/work2/largeModels/02-训练/LLaMA-Factory`

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（推荐 Ubuntu 22.04+） |
| Python | **≥ 3.11 正式版**（不可用 3.10 或 3.11.0rc1） |
| GPU | 推荐 NVIDIA GPU + CUDA（CPU 仅可调试） |
| 磁盘 | 模型 + 数据集 + checkpoint，建议 ≥ 50 GB |
| 内存 | 建议 ≥ 16 GB |

核心依赖版本（来自 `pyproject.toml`）：

| 包 | 版本 |
|----|------|
| torch | ≥ 2.4.0 |
| transformers | ≥ 4.55.0, ≤ 5.6.0 |
| peft | ≥ 0.18.0 |
| gradio | ≥ 4.38.0 |

---

## 2. 安装 Python 3.11（Ubuntu 22.04）

Ubuntu 22.04 默认只有 Python 3.10，且官方源中的 `python3.11` 可能是 **3.11.0rc1 候选版**，与 PyTorch 2.12 不兼容。必须通过 **deadsnakes PPA** 安装正式版。

### 2.1 添加 PPA 并安装

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

### 2.2 验证 Python 版本

```bash
python3.11 --version
# 正确：Python 3.11.15（或其他 3.11.x 正式版）
# 错误：Python 3.11.0rc1

python3.11 -c "import sys; print(hasattr(sys, 'get_int_max_str_digits'))"
# 必须输出 True
```

> **重要**：`sys.get_int_max_str_digits` 仅在 Python 3.11 正式版中存在。若为 `False` 或版本显示 `rc1`，PyTorch / transformers 导入会失败。

### 2.3 修复 dpkg 冲突（从 rc1 升级时）

若从 Ubuntu 自带 rc1 升级到 deadsnakes 3.11.15 时出现文件冲突：

```
trying to overwrite '/usr/lib/python3.11/sre_compile.py', which is also in package libpython3.11-minimal
```

按顺序执行：

```bash
# 强制覆盖 stdlib 包
sudo apt download libpython3.11-stdlib
sudo dpkg --force-overwrite -i libpython3.11-stdlib_*.deb

# 按依赖顺序配置
sudo dpkg --configure libpython3.11-minimal
sudo dpkg --configure libpython3.11-stdlib
sudo dpkg --configure -a
sudo apt --fix-broken install -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

若仍失败，彻底卸载后重装：

```bash
sudo apt remove --purge -y \
  python3.11 python3.11-minimal python3.11-dev python3.11-venv \
  libpython3.11 libpython3.11-dev libpython3.11-minimal libpython3.11-stdlib

sudo apt autoremove -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

确认包状态正常（应为 `ii`，不应有 `iU`）：

```bash
dpkg -l | grep python3.11
```

---

## 3. 创建虚拟环境

**必须使用虚拟环境**，避免与系统 Python 3.10 混用。

```bash
cd ~/work2/largeModels/02-训练/LLaMA-Factory

python3.11 -m venv .venv
source .venv/bin/activate

# 确认虚拟环境内的 Python 正确
python --version          # Python 3.11.x
which python              # .../LLaMA-Factory/.venv/bin/python
```

每次使用前激活：

```bash
source ~/work2/largeModels/02-训练/LLaMA-Factory/.venv/bin/activate
```

---

## 4. 安装 LLaMA Factory

### 4.1 升级 pip 和构建工具

旧版系统 pip（22.x）会导致 `packaging.licenses` 构建错误，必须先升级：

```bash
pip install -U pip setuptools wheel packaging hatchling
```

### 4.2 安装项目

```bash
cd ~/work2/largeModels/02-训练/LLaMA-Factory
pip install -e .
```

### 4.3 可选依赖

按需安装：

```bash
# 评测指标（BLEU/ROUGE）
pip install -r requirements/metrics.txt

# QLoRA 4/8 bit 量化训练
pip install -r requirements/bitsandbytes.txt

# DeepSpeed 多卡分布式
pip install -r requirements/deepspeed.txt

# vLLM 推理加速
pip install -r requirements/vllm.txt

# SwanLab 实验追踪
pip install -r requirements/swanlab.txt
```

### 4.4 GPU 版 PyTorch（若 CUDA 不可用）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

---

## 5. 验证安装

```bash
source .venv/bin/activate

llamafactory-cli version
# 期望输出：Welcome to LLaMA Factory, version 0.9.x

llamafactory-cli env
# 查看 CUDA、依赖版本等信息

llamafactory-cli help
# 查看所有子命令
```

---

## 6. 环境变量配置

### 6.1 模型/数据集下载加速（国内推荐）

```bash
# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用魔搭社区
export USE_MODELSCOPE_HUB=1
```

可写入 `~/.bashrc` 持久化：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 6.2 Web UI 监听地址

默认监听 `0.0.0.0`（所有网卡），可通过环境变量调整：

```bash
export GRADIO_SERVER_NAME=0.0.0.0    # 监听所有网卡（局域网可访问）
export GRADIO_SERVER_PORT=7860       # 端口
```

启动：

```bash
llamafactory-cli webui
# 局域网访问：http://<服务器IP>:7860
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GRADIO_SERVER_NAME` | `0.0.0.0` | 监听地址 |
| `GRADIO_SERVER_PORT` | `7860` | 端口 |
| `GRADIO_SHARE` | 未设置 | 设为 `1` 生成 Gradio 公网临时链接 |
| `GRADIO_IPV6` | 未设置 | 设为 `1` 使用 IPv6（`[::]`） |

外网访问需开放防火墙：

```bash
sudo ufw allow 7860/tcp
```

### 6.3 API 服务

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export API_KEY=sk-your-key        # 可选鉴权

llamafactory-cli api
```

### 6.4 训练相关

| 变量 | 说明 |
|------|------|
| `CUDA_VISIBLE_DEVICES` | 指定 GPU，如 `0,1` |
| `WANDB_DISABLED` | 设为 `true` 禁用 W&B |
| `FORCE_TORCHRUN` | 设为 `1` 强制多卡分布式 |
| `NPROC_PER_NODE` | 每节点 GPU 进程数 |

完整环境变量参考项目根目录 `.env.local` 文件。

---

## 7. 快速验证运行

安装完成后，按以下顺序验证各功能：

```bash
source .venv/bin/activate
cd ~/work2/largeModels/02-训练/LLaMA-Factory

# 1. Web UI
llamafactory-cli webui
# 浏览器打开 http://localhost:7860

# 2. 训练（示例配置，首次会下载模型）
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml

# 3. 对话测试（需先完成训练）
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml

# 4. API 服务
llamafactory-cli api examples/inference/qwen3_lora_sft.yaml
```

详细使用说明见 [10-usage-guide.md](./10-usage-guide.md)。

---

## 8. Docker 方式（可选）

若本地环境配置困难，可使用 Docker：

```bash
cd ~/work2/largeModels/02-训练/LLaMA-Factory/docker/docker-cuda
docker compose up -d
docker compose exec llamafactory bash

# 容器内直接使用
llamafactory-cli webui
```

或使用官方镜像：

```bash
docker run -it --rm --gpus=all --ipc=host \
  -p 7860:7860 -p 8000:8000 \
  hiyouga/llamafactory:latest
```

---

## 9. 常见问题排查

### Q1：`ModuleNotFoundError: No module named 'packaging.licenses'`

**原因**：系统 pip 过旧（22.x），隔离构建环境不兼容。

**解决**：

```bash
pip install -U pip packaging hatchling
pip install -e . --no-build-isolation   # 临时方案
# 推荐：使用 Python 3.11 虚拟环境 + 升级 pip 后正常安装
```

### Q2：`AttributeError: module 'sys' has no attribute 'get_int_max_str_digits'`

**原因**：Python 版本为 **3.11.0rc1** 候选版，不是正式版。

**解决**：按本文档 [§2](#2-安装-python-311ubuntu-2204) 安装 deadsnakes 正式版，重建虚拟环境。

### Q3：`Could not import module 'AutoModel'`

**原因**：通常是 Q2 的连锁反应（PyTorch 导入失败导致 transformers 不可用）。

**解决**：修复 Python 版本后重建 `.venv` 并重新 `pip install -e .`。

### Q4：CUDA OOM（显存不足）

**解决**：

- 使用 QLoRA：`examples/train_qlora/qwen3_lora_sft.yaml`
- 减小 `per_device_train_batch_size`
- 增大 `gradient_accumulation_steps`
- 使用 DeepSpeed ZeRO-3

### Q5：`Can't pickle local object`

**解决**：YAML 配置中设置 `dataloader_num_workers: 0`。

### Q6：Web UI 局域网无法访问

**检查**：

```bash
echo $GRADIO_SERVER_NAME    # 不应为 127.0.0.1
ss -tlnp | grep 7860        # 应显示 0.0.0.0:7860
sudo ufw allow 7860/tcp     # 开放防火墙
```

### Q7：模型下载超时

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或
export USE_MODELSCOPE_HUB=1
```

---

## 10. 完整安装脚本（一键参考）

以下脚本汇总上述步骤，可在新机器上逐步执行：

```bash
#!/bin/bash
set -e

# 1. 安装 Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 2. 验证
python3.11 --version
python3.11 -c "import sys; assert hasattr(sys, 'get_int_max_str_digits'), 'Python 3.11 rc1 detected!'"

# 3. 创建虚拟环境
cd ~/work2/largeModels/02-训练/LLaMA-Factory
python3.11 -m venv .venv
source .venv/bin/activate

# 4. 安装
pip install -U pip setuptools wheel packaging hatchling
pip install -e .
pip install -r requirements/metrics.txt

# 5. 配置镜像（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 6. 验证
llamafactory-cli version
llamafactory-cli env

echo "安装完成！运行 llamafactory-cli webui 启动 Web UI"
```

---

## 11. 下一步

环境就绪后：

| 目标 | 文档 |
|------|------|
| 快速上手训练/推理 | [10-usage-guide.md](./10-usage-guide.md) |
| Web UI 操作 | [09-webui.md](./09-webui.md) |
| YAML 配置说明 | [04-config-system.md](./04-config-system.md) |
| 理解训练流程 | [05-training-pipeline.md](./05-training-pipeline.md) |
