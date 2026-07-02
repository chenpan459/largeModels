# 在 Project Gutenberg 数据集上预训练 GPT

本目录包含在 Project Gutenberg 提供的免费书籍上训练小型 GPT 模型的代码。

正如 Project Gutenberg 网站所述：「绝大多数 Project Gutenberg 电子书在美国属于公有领域。」

使用 Project Gutenberg 资源前，请阅读 [Project Gutenberg 权限、许可与其他常见请求](https://www.gutenberg.org/policy/permission.html) 页面。

&nbsp;
## 如何使用本代码

&nbsp;

### 1）下载数据集

本节使用 [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub 仓库中的代码，从 Project Gutenberg 下载书籍。

截至编写时，大约需要 50 GB 磁盘空间，耗时约 10–15 小时；具体取决于 Project Gutenberg 此后增长了多少。

&nbsp;
#### Linux 与 macOS 用户下载说明

Linux 与 macOS 用户可按以下步骤下载数据集（Windows 用户请参阅下方说明）：

1. 将 `03_bonus_pretraining_on_gutenberg` 文件夹设为工作目录，以便在本目录下克隆 `gutenberg` 仓库（运行 `prepare_dataset.py` 和 `pretraining_simple.py` 时需要）。例如，在 `LLMs-from-scratch` 仓库根目录下执行：
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 克隆 `gutenberg` 仓库：
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 进入本地克隆的 `gutenberg` 仓库目录：
```bash
cd gutenberg
```

4. 在 `gutenberg` 仓库目录中安装 *requirements.txt* 中定义的依赖：
```bash
pip install -r requirements.txt
```

5. 下载数据：
```bash
python get_data.py
```

6. 返回 `03_bonus_pretraining_on_gutenberg` 文件夹：
```bash
cd ..
```

&nbsp;
#### Windows 用户特别说明

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 代码兼容 Linux 与 macOS。Windows 用户需做少量调整，例如在 `subprocess` 调用中添加 `shell=True`，并将 `rsync` 替换为等价方案。

在 Windows 上更简便的方式是使用「Windows Subsystem for Linux」（WSL），在 Windows 中运行 Ubuntu 等 Linux 环境。详见 [Microsoft 官方安装说明](https://learn.microsoft.com/en-us/windows/wsl/install) 与 [教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。

使用 WSL 时，请确保已安装 Python 3（通过 `python3 --version` 检查，或例如 `sudo apt-get install -y python3.10` 安装 Python 3.10），并安装以下包：

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **说明：**
> Python 环境与包安装说明见 [可选 Python 环境配置](../../setup/01_optional-python-setup-preferences/README.md) 与 [安装 Python 库](../../setup/02_installing-python-libraries/README.md)。
>
> 本仓库还提供运行 Ubuntu 的 Docker 镜像。使用说明见 [可选 Docker 环境](../../setup/03_optional-docker-environment/README.md)。

&nbsp;
### 2）准备数据集

接下来运行 `prepare_dataset.py`，将（截至编写时约 60,173 个）文本文件合并为更少、更大的文件，便于传输与访问：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **提示：**
> 生成的文件为纯文本格式，为简化起见未预先分词。若需多次使用数据集或训练多个 epoch，可改为保存已分词数据以节省计算时间。详见本页底部 *设计决策与改进*。

> **提示：**
> 可选择更小的文件大小，例如 50 MB。文件会更多，但适合在少量文件上快速试跑预训练。


&nbsp;
### 3）运行预训练脚本

可按如下方式运行预训练脚本。下列命令行参数以默认值展示，便于理解：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出格式大致如下：

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...


&nbsp;
> **提示：**
> 在 macOS 或 Linux 上，建议用 `tee` 将日志同时写入 `log.txt` 并在终端显示：

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **警告：**
> 在 V100 GPU 上，`gutenberg_preprocessed` 文件夹中约 500 MB 的单个文本文件预训练大约需要 4 小时。
> 该文件夹约有 47 个文件，全部跑完约需 200 小时（超过一周）。可先对少量文件试跑。


&nbsp;
## 设计决策与改进

本代码以教学为目的，保持简单、最小化。以下改进可提升建模效果与训练效率：

1. 修改 `prepare_dataset.py`，从每本书中去除 Gutenberg 固定页眉页脚等 boilerplate 文本。
2. 更新数据准备与加载工具，预先分词并保存，避免每次运行预训练脚本时重复分词。
3. 更新 `train_model_simple`，加入 [附录 D：为训练循环添加增强功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中的余弦衰减、线性 warmup 与梯度裁剪。
4. 更新预训练脚本以保存优化器状态（见第 5 章 *5.4 在 PyTorch 中加载与保存权重*；[ch05_ch.ipynb](../01_main-chapter-code/ch05_ch.ipynb)），并支持加载已有模型与优化器 checkpoint，在中断后继续训练。
5. 接入更完善的日志工具（如 Weights and Biases），实时查看损失与验证曲线。
6. 加入分布式数据并行（DDP），在多 GPU 上训练（见附录 A *A.9.3 多 GPU 训练*；[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。
7. 将 `previous_chapter.py` 中从零实现的 `MultiheadAttention` 替换为 [高效多头注意力实现](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) 中的 `MHAPyTorchScaledDotProduct`，通过 PyTorch 的 `nn.functional.scaled_dot_product_attention` 使用 Flash Attention。
8. 通过 [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)（`model = torch.compile`）或 [thunder](https://github.com/Lightning-AI/lightning-thunder)（`model = thunder.jit(model)`）编译模型以加速训练。
9. 实现 Gradient Low-Rank Projection（GaLore）进一步加速预训练：将 `AdamW` 替换为 [GaLore Python 库](https://github.com/jiaweizzhao/GaLore) 提供的 `GaLoreAdamW` 即可。
