#!/bin/bash
# 将 gitlink/子仓库 转为普通源码目录（移除嵌套 .git 后由父仓库跟踪）
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"

REPOS=(
  "01-模型原理/nanoGPT"
  "01-模型原理/LLMs-from-scratch"
  "02-训练/how-to-train-your-gpt"
  "02-训练/LLaMA-Factory"
  "03-推理部署/llama.cpp"
  "03-推理部署/vllm"
  "04-量化内核/ggml"
  "05-RAG/llama_index"
  "06-Research/beyond-nanogpt"
  "06-Research/Megatron-LM"
)

echo ">>> 从索引移除 gitlink ..."
for r in "${REPOS[@]}"; do
  if git ls-files --stage "$r" 2>/dev/null | grep -q "^160000"; then
    git rm --cached -f "$r"
    echo "  removed gitlink: $r"
  fi
done

echo ">>> 移除嵌套 .git ..."
find . -path "./.git" -prune -o -name ".git" -type d -print | while read -r g; do
  echo "  rm -rf $g"
  rm -rf "$g"
done

echo ">>> 完成。请执行: git add -A && git status"
