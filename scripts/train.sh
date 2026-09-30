#!/usr/bin/env bash
# =============================================================================
# DeepSearch-RL 训练启动脚本（4×5090 / veRL / SGLang / GRPO）
#
# 用法：
#   bash scripts/train.sh                       # 用默认 configs/grpo_qwen3_8b_4x5090.yaml
#   bash scripts/train.sh data.train_batch_size=128   # 后面的 key=value 直接透传 Hydra
#
# 前置（另外两个终端里要先起好）：
#   - 检索服务：  bash scripts/start_retrieval.sh     (默认 127.0.0.1:8000)
#   - Judge 服务：bash scripts/start_judge.sh          (默认 127.0.0.1:8001)
#
# 必填环境变量：
#   MODEL_PATH        本地 Qwen3-8B 权重目录（先跑 scripts/download_model.sh）
#   SWANLAB_API_KEY   云端追踪用（不想上传可 export SWANLAB_MODE=offline）
# =============================================================================
set -euo pipefail

# 切到工程根目录（本脚本位于 scripts/ 下）
cd "$(dirname "$0")/.."

# ---- 1) 关键环境变量检查（只提醒，不强行 export，便于用户自定义）----
if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "[警告] 未设置 MODEL_PATH。"
    echo "       请先执行： bash scripts/download_model.sh"
    echo "       然后再：   export MODEL_PATH=\$HOME/models/Qwen3-8B"
    echo "       （也可以直接在命令行覆盖： MODEL_PATH=/xxx bash scripts/train.sh）"
    echo ""
fi

if [[ -z "${SWANLAB_API_KEY:-}" && "${SWANLAB_MODE:-cloud}" != "offline" ]]; then
    echo "[提示] 未设置 SWANLAB_API_KEY，且 SWANLAB_MODE 不是 offline。"
    echo "       veRL 会按 trainer.logger=[console,swanlab] 初始化 swanlab；"
    echo "       云端模式需要 key，否则 swanlab.init 可能失败（不影响训练本身）。"
    echo "       本地/无网环境可先： export SWANLAB_MODE=offline"
    echo ""
fi

echo "[info] cwd=$(pwd)"
echo "[info] MODEL_PATH=${MODEL_PATH:-<未设置>}"
echo "[info] SWANLAB_MODE=${SWANLAB_MODE:-cloud}"

# ---- 2) 启动训练（剩余参数原样透传）----
exec python -m deepsearch_rl.train.train_grpo \
    --config configs/grpo_qwen3_8b_4x5090.yaml \
    "$@"
