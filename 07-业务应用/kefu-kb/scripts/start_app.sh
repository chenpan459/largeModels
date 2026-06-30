#!/bin/bash
# 启动客服知识库 API + Web UI
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt -q
fi

export PYTHONPATH=.
.venv/bin/python -m app.main
