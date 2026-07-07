# 客服知识库 (kefu-kb)

基于 **RAG** 的智能客服系统：文档知识库 + 向量检索 + llama-server 生成回答。

## 架构

```
用户提问 (Web UI)
    |
    v
FastAPI /api/ask
    |
    +-- Qdrant 向量检索 (embedding)
    +-- llama-server rerank (可选)
    +-- llama-server chat (生成回答)
    |
    v
回答 + 引用来源
```

## 依赖服务

| 服务 | 用途 | 启动方式 |
|------|------|----------|
| **Qdrant** | 向量数据库 | `docker compose up -d` |
| **llama-server** | Embedding + Chat (+ Rerank) | 见下方 |

### llama-server 示例

需要至少 **chat 模型**；推荐同时启动 **embedding** 和 **rerank** 模型。

```bash
# 终端 1: Chat 模型 (按你的 GGUF 路径修改)
cd ../../03-推理部署/llama.cpp
./build/bin/llama-server -m /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080

# 终端 2: Embedding (可选，与 chat 同端口需合并或改 config.yaml 端口)
# ./build/bin/llama-server -m /path/to/bge-m3.gguf --embedding --port 8081
```

若 embedding/chat 在不同端口，修改 `config.yaml` 中 `llama.base_url`，或部署多模型 router。

## 快速开始

```bash
cd /home/cp/work2/largeModels/07-业务应用/kefu-kb

# 1. 启动 Qdrant
docker compose up -d

# 2. 安装 Python 依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. 确保 llama-server 已运行在 config.yaml 配置的地址

# 4. 文档入库
export PYTHONPATH=.
python -c "from app.ingest import KnowledgeStore; s=KnowledgeStore(); f,c=s.ingest_directory('data/docs'); print(f'files={f} chunks={c}')"

# 5. 启动应用
python -m app.main
```

浏览器打开: http://127.0.0.1:8000

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 界面（问答 + 知识库管理） |
| `/api/health` | GET | 健康检查 |
| `/api/docs` | GET | 列出知识库文档 |
| `/api/docs/upload` | POST | 上传文档（multipart） |
| `/api/docs/{path}` | DELETE | 删除文档 |
| `/api/ingest` | POST | 重新索引 `data/docs/` |
| `/api/search` | POST | 仅检索 `{"question":"..."}` |
| `/api/ask` | POST | 问答 `{"question": "..."}` |

### 问答示例

```bash
curl -s http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"七天无理由退货条件是什么？"}' | python3 -m json.tool
```

## 知识库管理

1. **Web UI**：打开「知识库管理」Tab，上传 `.md` / `.txt` 文件，点击「重新入库」
2. **手动放置**：将文档放入 `data/docs/`，调用 `POST /api/ingest`
3. 内置示例：`faq.md`、`return_policy.md`、`membership.md`、`shipping.md`

### 上传示例

```bash
curl -F "file=@data/docs/faq.md" http://127.0.0.1:8000/api/docs/upload
curl -X POST http://127.0.0.1:8000/api/ingest
```

### 仅检索（不调 LLM）

```bash
curl -s http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"question":"运费怎么算？"}' | python3 -m json.tool
```

## 配置

编辑 `config.yaml`:

- `llama.base_url` - llama-server 地址
- `rag.chunk_size` - 分块大小
- `rag.use_rerank` - 是否启用 rerank（需 server 支持）

## 目录结构

```
kefu-kb/
├── app/
│   ├── main.py          # FastAPI 入口
│   ├── ingest.py        # 文档分块 + 向量化入库
│   ├── retriever.py     # 检索 + rerank
│   ├── chat.py          # 客服 Prompt + 生成
│   └── llama_client.py  # llama-server API 客户端
├── data/docs/           # 知识库文档（示例：FAQ、退换货、会员）
├── static/index.html    # 客服 Web UI
├── config.yaml
└── docker-compose.yml   # Qdrant
```

## 生产化建议

- 增加用户鉴权、会话历史、人工转接
- 按产品线/租户划分 collection
- 接入企微/钉钉/网页插件
- 建立测试集定期评估检索命中率
- 敏感词过滤与回答审核
