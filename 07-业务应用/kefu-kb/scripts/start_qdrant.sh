#!/bin/bash
# 启动 Qdrant 向量库
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose up -d
echo "Qdrant: http://127.0.0.1:6333/dashboard"
