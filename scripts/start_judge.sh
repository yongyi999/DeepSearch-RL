#!/usr/bin/env bash
# 启动远程 vLLM Judge 服务（OpenAI 兼容端点，默认端口 8001）。
# 用法：
#   bash scripts/start_judge.sh
# 可通过环境变量覆盖默认值：
#   JUDGE_MODEL=Qwen/Qwen3-8B JUDGE_PORT=8001 JUDGE_TP=1 JUDGE_GPU_MEM=0.4 bash scripts/start_judge.sh
set -euo pipefail

# ---- 可被环境变量覆盖的参数 ----
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_TP="${JUDGE_TP:-1}"
JUDGE_GPU_MEM="${JUDGE_GPU_MEM:-0.4}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"

# 优先用工程内的便捷启动器（等价于下面的手动 vllm 命令）。
exec python -m deepsearch_rl.judge.judge_server \
  --model "${JUDGE_MODEL}" \
  --port "${JUDGE_PORT}" \
  --tensor-parallel-size "${JUDGE_TP}" \
  --gpu-memory-utilization "${JUDGE_GPU_MEM}" \
  --max-model-len "${JUDGE_MAX_MODEL_LEN}"

# ---- 等价的手动 vllm serve 命令（可直接复制运行）----
# vllm serve "${JUDGE_MODEL}" \
#   --served-model-name judge \
#   --host 0.0.0.0 --port "${JUDGE_PORT}" \
#   --tensor-parallel-size "${JUDGE_TP}" \
#   --gpu-memory-utilization "${JUDGE_GPU_MEM}" \
#   --max-model-len "${JUDGE_MAX_MODEL_LEN}" \
#   --dtype bfloat16 \
#   --enable-prefix-caching
#
# 启动后客户端连接：
#   export JUDGE_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
#   export JUDGE_MODEL="judge"   # 与 --served-model-name 一致
