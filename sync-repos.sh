#!/bin/bash
# 同步 largeModels 学习项目仓库（按分类目录）
# 用法: ./sync-repos.sh
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
GIT=(git -c http.proxy= -c https.proxy= -c http.https://github.com.proxy=)

# 分类目录
DIR_01="$BASE/01-模型原理"
DIR_02="$BASE/02-训练"
DIR_03="$BASE/03-推理部署"
DIR_04="$BASE/04-量化内核"
DIR_05="$BASE/05-RAG"
DIR_06="$BASE/06-Research"

mkdir -p "$DIR_01" "$DIR_02" "$DIR_03" "$DIR_04" "$DIR_05" "$DIR_06"

clone_repo() {
  local parent="$1" name="$2" url="$3"
  local dir="$parent/$name"
  if [[ -d "$dir/.git" ]]; then
    echo "SKIP clone (exists): $dir"
    return 0
  fi
  local owner repo
  owner=$(echo "$url" | sed -n 's#.*github.com/\([^/]*\)/\([^/.]*\).*#\1#p')
  repo=$(echo "$url" | sed -n 's#.*github.com/\([^/]*\)/\([^/.]*\).*#\2#p')
  local mirrors=(
    "$url"
    "https://gitclone.com/github.com/${owner}/${repo}"
    "https://mirror.ghproxy.com/https://github.com/${owner}/${repo}.git"
  )
  local extra=()
  [[ "$name" == "Megatron-LM" || "$name" == "vllm" || "$name" == "llama_index" ]] && extra+=(--filter=blob:none)

  rm -rf "$dir" 2>/dev/null || true
  for u in "${mirrors[@]}"; do
    echo "Cloning $name -> $dir from $u ..."
    if "${GIT[@]}" clone --depth 1 --single-branch "${extra[@]}" "$u" "$dir" 2>&1; then
      echo "OK: $dir"
      return 0
    fi
  done
  echo "FAIL: $dir"
  return 1
}

pull_repo() {
  local dir="$1"
  if [[ ! -d "$dir/.git" ]]; then
    echo "SKIP pull (no .git): $dir"
    return 0
  fi
  echo "Pulling $dir ..."
  (cd "$dir" && "${GIT[@]}" pull --ff-only 2>&1) && echo "Pulled $dir" || echo "Pull failed/skipped: $dir"
}

cd "$BASE"

echo "========== 1. 模型原理 =========="
clone_repo "$DIR_01" nanoGPT "https://github.com/karpathy/nanoGPT.git" || true
clone_repo "$DIR_01" LLMs-from-scratch "https://github.com/rasbt/LLMs-from-scratch.git" || true

echo "========== 2. 训练 =========="
clone_repo "$DIR_02" how-to-train-your-gpt "https://github.com/obarannikov/how-to-train-your-gpt.git" || true
clone_repo "$DIR_02" LLaMA-Factory "https://github.com/hiyouga/LLaMA-Factory.git" || true

echo "========== 3. 推理部署 =========="
clone_repo "$DIR_03" llama.cpp "https://github.com/ggml-org/llama.cpp.git" || true
clone_repo "$DIR_03" vllm "https://github.com/vllm-project/vllm.git" || true

echo "========== 4. 量化/内核 =========="
clone_repo "$DIR_04" ggml "https://github.com/ggml-org/ggml.git" || true

echo "========== 5. RAG =========="
clone_repo "$DIR_05" llama_index "https://github.com/run-llama/llama_index.git" || true

echo "========== 6. Research =========="
clone_repo "$DIR_06" beyond-nanogpt "https://github.com/tanishqkumar/beyond-nanogpt.git" || true
clone_repo "$DIR_06" Megatron-LM "https://github.com/NVIDIA/Megatron-LM.git" || true

echo "========== 更新已有仓库 =========="
for d in \
  "$DIR_01/nanoGPT" \
  "$DIR_01/LLMs-from-scratch" \
  "$DIR_02/how-to-train-your-gpt" \
  "$DIR_02/LLaMA-Factory" \
  "$DIR_03/llama.cpp" \
  "$DIR_03/vllm" \
  "$DIR_04/ggml" \
  "$DIR_05/llama_index" \
  "$DIR_06/beyond-nanogpt" \
  "$DIR_06/Megatron-LM"
do
  pull_repo "$d"
done

echo ""
printf "%-40s %-8s %s\n" "PATH" "STATUS" "COMMIT"
printf "%-40s %-8s %s\n" "----" "------" "------"
ALL=(
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
for r in "${ALL[@]}"; do
  d="$BASE/$r"
  if [[ -d "$d/.git" ]]; then
    h=$(cd "$d" && "${GIT[@]}" rev-parse --short HEAD 2>/dev/null || echo "?")
    printf "%-40s %-8s %s\n" "$r" "OK" "$h"
  else
    printf "%-40s %-8s %s\n" "$r" "MISSING" "-"
  fi
done

echo ""
echo "注: llama-server 位于 03-推理部署/llama.cpp/tools/server/"
echo "注: llama.cppDoc 位于 03-推理部署/llama.cppDoc/"
