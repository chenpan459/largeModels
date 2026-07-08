# OCR — baidu/Unlimited-OCR

基于百度 [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) 的文档解析服务，支持单图、多页图片和 PDF。

```
图片 / PDF
    → Unlimited-OCR (3B, vision-language)
    → Markdown / 结构化文本
    → CLI 或 Web API 输出
```

## 前置条件

| 项目 | 要求 |
|------|------|
| Python | **3.11+**（官方测试 3.12 + CUDA 12.9） |
| GPU | **强烈推荐** NVIDIA GPU，BF16，显存 ≥ 8 GB |
| 磁盘 | ≥ 10 GB；**模型缓存建议放 `/home/work2`（根分区常满）** |
| 网络 | 首次运行从 Hugging Face 下载模型 |

## 快速开始

### 1. 安装依赖

```bash
cd ~/work2/largeModels/07-业务应用/ocr

# venv 也建在 work2 项目目录，避免占满根分区 /
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**磁盘说明**：`/home/cp` 在根分区 `/dev/sda3`（你当前约 **96% 满，仅 ~2GB 空闲**），而 `~/work2` 在 `/dev/sdc`（约 **47GB 空闲**）。`config.yaml` 已将 Hugging Face 模型缓存指向 `~/work2/largeModels/models/huggingface`。若 pip 安装仍报空间不足，可清理根分区缓存：

```bash
pip cache purge          # 约可释放 ~/.cache/pip 数 GB
```

若使用 GPU，请按 [PyTorch 官网](https://pytorch.org/) 安装对应 CUDA 版本的 `torch`。

### 2. 命令行 OCR

```bash
source .venv/bin/activate
export PYTHONPATH=.

# 单张图片（gundam 模式，适合复杂版面）
python -m app.cli image /path/to/page.jpg

# 单张图片（base 模式，整页）
python -m app.cli image /path/to/page.jpg --mode base

# PDF 多页解析
python -m app.cli pdf /path/to/document.pdf

# 多张图片
python -m app.cli images page1.png page2.png page3.png
```

### 3. 启动 Web 服务

```bash
source .venv/bin/activate
export PYTHONPATH=.

python -m app.main
```

| 地址 | 说明 |
|------|------|
| http://127.0.0.1:8010 | Web 上传界面 |
| http://127.0.0.1:8010/api/health | 健康检查 |

### 4. API 示例

```bash
# 单图 OCR
curl -X POST http://127.0.0.1:8010/api/ocr/image \
  -F "file=@page.jpg" \
  -F "mode=gundam"

# PDF OCR
curl -X POST http://127.0.0.1:8010/api/ocr/pdf \
  -F "file=@document.pdf"
```

## 解析模式

| 模式 | 适用 | 参数 |
|------|------|------|
| **gundam** | 单张复杂文档、表格、混合版面 | `base_size=1024, image_size=640, crop_mode=True` |
| **base** | 整页扫描、多页/PDF | `image_size=1024, crop_mode=False` |

多页和 PDF 固定使用 **base** 模式（与官方文档一致）。

## 配置

编辑 `config.yaml`：

```yaml
model:
  name: baidu/Unlimited-OCR
  device: auto          # auto | cuda | cpu
  dtype: bfloat16

single_image:
  mode: gundam
  prompt: "<image>document parsing."

multi_page:
  prompt: "<image>Multi page parsing."

server:
  port: 8010
```

## 目录结构

```
ocr/
├── app/
│   ├── ocr_engine.py    # 模型封装
│   ├── pdf_utils.py     # PDF 转图片
│   ├── cli.py           # 命令行
│   └── main.py          # FastAPI 服务
├── static/index.html    # Web UI
├── config.yaml
├── outputs/             # OCR 结果（自动生成）
├── data/uploads/        # 上传临时文件
└── requirements.txt
```

## 与其他部署方式

| 方式 | 说明 |
|------|------|
| 本项目 | Transformers 本地推理，适合集成到业务 |
| [vLLM Recipe](https://recipes.vllm.ai/baidu/Unlimited-OCR) | 高吞吐 GPU 服务 |
| [Hugging Face 官方示例](https://huggingface.co/baidu/Unlimited-OCR) | 原始 `model.infer()` 用法 |

## 常见问题

| 现象 | 处理 |
|------|------|
| 首次运行很慢 | 正在下载 ~6GB 模型，请耐心等待 |
| CUDA out of memory | 换更大显存 GPU，或 `device: cpu`（极慢） |
| 输出为空 | 查看 `outputs/<name>/` 下的 `.md` / `.txt` 文件 |
| `trust_remote_code` 报错 | 确保 `transformers>=4.57.0` |

## 引用

```bibtex
@misc{yin2026unlimitedocrworks,
      title={Unlimited OCR Works},
      author={Youyang Yin and others},
      year={2026},
      eprint={2606.23050},
}
```
