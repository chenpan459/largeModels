#!/bin/bash
# 将 data/docs 文档入库
set -euo pipefail
cd "$(dirname "$0")/.."
PORT=$(grep -A2 '^server:' config.yaml | grep port | awk '{print $2}')
curl -s -X POST "http://127.0.0.1:${PORT}/api/ingest" | python3 -m json.tool
