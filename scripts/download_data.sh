#!/usr/bin/env bash
# ============================================================
# DeepSearch-RL 数据一键下载 / 评测集构建 / 训练 parquet 预处理
# 面向 Ubuntu + bash（AutoDL 4xRTX5090）。
#
# 用法：
#   bash scripts/download_data.sh
#
# 可选环境变量：
#   HF_ENDPOINT   国内 HF 镜像，默认 https://hf-mirror.com
#   MODELSCOPE_CACHE  可选：指定 modelscope 缓存目录
# ============================================================
set -euo pipefail

# 切到工程根目录（本脚本在 scripts/ 下）
cd "$(dirname "$0")/.."

# ---- 0. 环境与依赖提示 ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "[env] HF_ENDPOINT=${HF_ENDPOINT}"

if ! python -c "import datasets" 2>/dev/null; then
  echo "[提示] 未检测到 datasets，建议先安装："
  echo "  pip install -U datasets huggingface_hub modelscope pyarrow"
fi

# 统一输出目录（可通过环境变量覆盖）
RAW_DIR="${RAW_DIR:-data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
EVAL_OUT="${EVAL_OUT:-data/eval_hard_500.jsonl}"

# ---- 1. 下载 + 归一为统一中间 jsonl ----
#   --nq_limit 30000：下载期对 NQ-open train 的硬截断（省磁盘）
#   --backend auto  ：优先 modelscope，失败回退 HF datasets（自动走 HF_ENDPOINT 镜像）
echo ">>> [1/3] 下载并归一数据集 -> ${RAW_DIR}"
python data/download_data.py \
  --out_dir "${RAW_DIR}" \
  --sources nq,hotpotqa,2wiki,musique,bamboogle \
  --nq_limit 30000 \
  --backend auto

# ---- 2. 构建 500 题冻结评测集（seed=42）----
echo ">>> [2/3] 构建 500 题冻结评测集 -> ${EVAL_OUT}"
python data/build_eval_500.py \
  --raw_dir "${RAW_DIR}" \
  --out "${EVAL_OUT}" \
  --seed 42

# ---- 3. 生成 veRL 训练 parquet（NQ 再按 seed=42 降采样到 30000）----
echo ">>> [3/3] 生成 veRL 训练/val parquet -> ${PROCESSED_DIR}"
python data/prepare_train.py \
  --raw_dir "${RAW_DIR}" \
  --out_dir "${PROCESSED_DIR}" \
  --nq_limit 30000 \
  --seed 42

echo ">>> 全部完成。"
echo "    中间数据 : ${RAW_DIR}"
echo "    训练 parquet : ${PROCESSED_DIR}/train/  与  ${PROCESSED_DIR}/val/"
echo "    评测集   : ${EVAL_OUT}"
