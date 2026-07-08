# 07 - 业务应用

面向业务落地的 LLM 项目，与 `05-RAG/llama_index`（框架学习）互补。

## OCR 项目：ocr

路径：`/home/cp/work2/largeModels/07-业务应用/ocr`

基于 **[baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)** 的文档解析：单图 / 多页 / PDF → 结构化文本，提供 CLI 与 Web API。

```
图片或 PDF
    → Unlimited-OCR (Transformers)
    → Markdown / 文本
    → CLI 或 http://127.0.0.1:8010
```

| 项目 | 要求 |
|------|------|
| Python | 3.11+ |
| GPU | 推荐 NVIDIA ≥ 8GB 显存 |
| 磁盘 | ≥ 10 GB |

```bash
cd ~/work2/largeModels/07-业务应用/ocr
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
python -m app.cli image your.jpg
python -m app.main   # Web 服务 :8010
```

详见 **[ocr/README.md](ocr/README.md)**。

---

## 知识库项目：kefu-kb

路径：`/home/cp/work2/largeModels/07-业务应用/kefu-kb`

基于 **RAG** 的可运行知识库：文档入库 → 向量检索 → 大模型生成回答，附引用来源。

```
Markdown/TXT 文档
    → 分块 + Embedding（本地模型 或 llama-server）
    → Qdrant 向量库（本地文件 或 Docker）
    → 检索 (+ 可选 Rerank)
    → llama-server Chat 生成
    → 问答 + 引用来源
```

---

## 部署前置条件

| 项目 | 要求 |
|------|------|
| Python | **3.11+** 正式版 |
| 磁盘 | 建议 ≥ 5 GB（含 embedding 模型缓存） |
| 网络 | 首次安装需下载 `sentence-transformers` 模型 |
| llama-server | **智能问答** 时需要；入库/检索可不需要 |
| Docker | **可选**（仅在使用 Docker 版 Qdrant 时需要） |

---

## 部署步骤（推荐：零 Docker）

### 1. 创建 Python 环境

```bash
cd ~/work2/largeModels/07-业务应用/kefu-kb

python3.11 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### 2. 确认配置

默认 `config.yaml` 已启用**本地模式**，无需 Docker、无需 llama-server 即可入库：

```yaml
embedding:
  backend: local                              # 本地 embedding
  local_model: paraphrase-multilingual-MiniLM-L12-v2

qdrant:
  local_path: "data/qdrant_storage"           # 本地向量库，无需 Docker

rag:
  use_rerank: false                           # 无 rerank 服务时可关闭

server:
  host: "0.0.0.0"                             # 局域网可访问
  port: 8000
```

### 3. 文档入库

```bash
source .venv/bin/activate
export PYTHONPATH=.

python -c "
from app.ingest import KnowledgeStore
s = KnowledgeStore()
f, c = s.ingest_directory('data/docs')
print(f'files={f} chunks={c}')
"
```

首次运行会下载 embedding 模型（约 400MB），成功输出类似 `files=4 chunks=xx`。

### 4. 启动 llama-server（智能问答必需）

入库和 `/api/search` 不依赖 llama-server；**Web 问答** `/api/ask` 需要 chat 模型：

```bash
cd ~/work2/largeModels/03-推理部署/llama.cpp

./build-vm/bin/llama-server \
  -m ~/models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 \
  -t 4 --parallel 4
```

将 `-m` 改为你本机 GGUF 路径，等到终端出现 `HTTP server is listening`。

验证：

```bash
curl http://127.0.0.1:8080/health
```

### 5. 启动知识库服务

```bash
cd ~/work2/largeModels/07-业务应用/kefu-kb
source .venv/bin/activate
export PYTHONPATH=.

python -m app.main
```

| 访问地址 | 说明 |
|----------|------|
| http://127.0.0.1:8000 | 本机 Web UI |
| http://\<服务器IP\>:8000 | 局域网访问（`host: 0.0.0.0`） |

Web UI 两个 Tab：

- **智能问答** — RAG 对话（需 llama-server）
- **知识库管理** — 上传文档 → 重新入库

---

## 服务依赖一览

| 组件 | 默认方式 | 用途 | 是否必须 |
|------|----------|------|----------|
| Python venv | 本地 | 运行 FastAPI | 是 |
| Embedding | `embedding.backend: local` | 文档向量化、检索 | 是 |
| Qdrant | `local_path` 本地文件 | 向量存储 | 是 |
| llama-server | `:8080` | Chat 生成回答 | 问答时必须 |
| Docker Qdrant | 可选 | 独立向量库服务 | 否 |
| Rerank | `use_rerank: false` | 精排检索结果 | 否 |

---

## 部署验证

```bash
# 健康检查
curl http://127.0.0.1:8000/api/health | python3 -m json.tool
# qdrant: true, points_count > 0 表示入库成功
# llama: true 表示 chat 服务可用

# 仅检索（无需 llama-server）
curl -s http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"question":"运费怎么算？"}' | python3 -m json.tool

# 智能问答（需 llama-server）
curl -s http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"七天无理由退货条件是什么？"}' | python3 -m json.tool
```

---

## 可选：Docker 版 Qdrant

若需独立 Qdrant 容器，在 `config.yaml` 中**注释** `local_path`，使用 `host/port`：

```bash
cd ~/work2/largeModels/07-业务应用/kefu-kb

# 不要用 sudo，确保当前用户在 docker 组
docker compose up -d
```

---

## 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `Connection refused :8080` | llama-server 未启动 | 入库用 `embedding.backend: local`；问答需启动 llama-server |
| `No such file: .venv/bin/activate` | 未创建虚拟环境 | `python3.11 -m venv .venv` |
| `permission denied docker.sock` | 无 Docker 权限 | 用本地 Qdrant（`local_path`），或 `sudo usermod -aG docker $USER` |
| Docker 拉镜像 timeout | 无法访问 Docker Hub | 用本地 Qdrant；或配置镜像加速 |
| `No space left on device` | 磁盘满 | `df -h` 清理空间后再 `pip install` |
| 问答返回「暂无相关信息」 | 未入库或问题不匹配 | 检查 `/api/health` 的 `points_count`，重新 ingest |
| `llama: false` 但检索正常 | 仅 chat 不可用 | 启动 llama-server；`/api/search` 仍可用 |

---

## 配置切换

| 场景 | config.yaml 修改 |
|------|------------------|
| 全本地开发（默认） | `embedding.backend: local` + `qdrant.local_path` |
| embedding 走 llama-server | `embedding.backend: llama` |
| 使用 Docker Qdrant | 注释 `local_path`，`docker compose up -d` |
| 启用 rerank | `rag.use_rerank: true`（需 llama-server 支持 `/v1/rerank`） |
| 改监听端口 | `server.port` |

---

## 目录结构

```
07-业务应用/
├── ocr/                   # Unlimited-OCR 文档解析
│   ├── app/               # CLI + FastAPI
│   ├── static/            # Web UI
│   └── outputs/           # OCR 结果
└── kefu-kb/
    ├── app/                 # FastAPI 应用
    ├── config.yaml          # 部署配置（embedding / qdrant / llama）
    ├── data/
    │   ├── docs/            # 知识库文档（Markdown/TXT）
    │   └── qdrant_storage/  # 本地向量库（自动生成）
    ├── static/index.html    # Web UI
    ├── docker-compose.yml   # 可选 Qdrant
    ├── requirements.txt
    └── README.md            # 项目详细文档
```

---

## 学习路径

```
03-推理部署 (llama-server / vLLM)
    ↓
07-业务应用 (ocr 文档解析 / kefu-kb 知识库)
    ↓
05-RAG (LlamaIndex 深入)
```

更多 API、知识库管理说明见 **[kefu-kb/README.md](kefu-kb/README.md)**。
