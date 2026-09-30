#!/usr/bin/env bash
# ============================================================
# DeepSearch-RL 模型下载：Qwen/Qwen3-8B（ModelScope）
#
# 用法：
#   bash scripts/download_model.sh
#
# 可选环境变量：
#   MODEL_DIR   模型落盘目录（默认 ~/models/Qwen3-8B）
#   MODEL_ID    ModelScope 模型 id（默认 Qwen/Qwen3-8B）
#
# 本脚本只给出两种等价方式，不在此真正发起下载（按需取消注释执行）。
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3-8B}"

echo "[info] MODEL_ID = ${MODEL_ID}"
echo "[info] MODEL_DIR = ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# 前置依赖
if ! python -c "import modelscope" 2>/dev/null; then
  echo "[提示] 未安装 modelscope，先执行： pip install -U modelscope"
fi

# ---- 方式 A：modelscope 命令行（推荐，断点续传）----
#   modelscope download \
#       --model "${MODEL_ID}" \
#       --local_dir "${MODEL_DIR}"

# ---- 方式 B：python SDK ----
# python - <<PY
# from modelscope import snapshot_download
# p = snapshot_download(
#     "${MODEL_ID}",
#     local_dir="${MODEL_DIR}",
# )
# print("模型已下载到：", p)
# PY

echo ">>> 取消上方任一方式的注释即可开始下载。"
echo ">>> 下载完成后，把 MODEL_PATH 指向 ${MODEL_DIR} 即可在 configs/grpo_qwen3_8b_4x5090.yaml 中使用。"
