#!/usr/bin/env bash
# =============================================================================
# DeepSearch-RL 500 题冻结评测启动脚本（Ubuntu + bash）
#
# 环境变量（均可覆盖默认值）：
#   MODEL_ENDPOINT   被评模型 SGLang/vLLM OpenAI 端点（默认 http://127.0.0.1:30000/v1）
#   MODEL_NAME       被评模型 served-model-name（默认 default）
#   JUDGE_BASE_URL   Judge 服务端点（默认 http://127.0.0.1:8001/v1）
#   JUDGE_MODEL      Judge 模型名（默认 judge）
#   SEARCH_BACKEND   搜索后端 serper/serpapi/bing/brave/tavily/ddg（默认 ddg）
#
# 用法：
#   bash scripts/eval.sh                       # 全量 500 题
#   bash scripts/eval.sh --limit 20             # 只跑前 20 题（调试）
#   MODEL_ENDPOINT=http://10.0.0.1:30000/v1 bash scripts/eval.sh
#   bash scripts/eval.sh --compare outputs/eval/metrics_baseline.json outputs/eval/metrics_trained.json
# =============================================================================
set -e

# 切到工程根目录（本脚本位于 <root>/scripts/ 下）
cd "$(dirname "$0")/.."

# 端点默认值
MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:30000/v1}"
MODEL_NAME="${MODEL_NAME:-default}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8001/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-judge}"
SEARCH_BACKEND="${SEARCH_BACKEND:-ddg}"

echo "[eval] model  = ${MODEL_ENDPOINT} (${MODEL_NAME})"
echo "[eval] judge   = ${JUDGE_BASE_URL} (${JUDGE_MODEL})"
echo "[eval] backend = ${SEARCH_BACKEND}"

# 把环境变量透传给 argparse（argparse 有自己的默认值，这里只在显式设置时覆盖）
ARGS=(
  --model_endpoint "${MODEL_ENDPOINT}"
  --model_name "${MODEL_NAME}"
  --judge_base_url "${JUDGE_BASE_URL}"
  --judge_model "${JUDGE_MODEL}"
  --search_backend "${SEARCH_BACKEND}"
)

# 其余参数（--limit/--out/--save_trajectories/--compare 等）原样透传
exec python -m deepsearch_rl.eval.evaluate "${ARGS[@]}" "$@"
