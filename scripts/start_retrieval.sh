#!/usr/bin/env bash
# =============================================================================
# 启动 DeepSearch-RL 检索服务（独立 FastAPI 进程）
#
# 用法：
#   bash scripts/start_retrieval.sh
#
# 常用环境变量（均有缺省值，可按需覆盖）：
#   SEARCH_BACKEND         搜索后端：serper/serpapi/bing/brave/tavily/ddg
#                          默认 serper；未配置任何 key 时服务会自动回退 ddg。
#   SEARCH_API_KEYS / SERPER_API_KEYS / SERPAPI_API_KEYS / BING_API_KEYS /
#   BRAVE_API_KEYS / TAVILY_API_KEYS   逗号分隔的 API key。
#   KEY_FILE               API key 文件（每行一个，# 开头为注释）。
#   RETRIEVAL_PORT         服务端口（默认 8000）。
#   RETRIEVAL_CONCURRENCY  每实例并发上限（默认 120）。
#   CACHE_DB_PATH          SQLite 缓存路径（默认 ~/.cache/deepsearch_rl/tool_cache.db）。
#   UVICORN_WORKERS        uvicorn worker 数（默认 1；多 worker 共享同一 SQLite 缓存）。
# =============================================================================
set -e

# ---- 配置（带缺省值）----
export SEARCH_BACKEND="${SEARCH_BACKEND:-serper}"
export RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
export RETRIEVAL_CONCURRENCY="${RETRIEVAL_CONCURRENCY:-120}"
export CACHE_DB_PATH="${CACHE_DB_PATH:-$HOME/.cache/deepsearch_rl/tool_cache.db}"
export KEY_FILE="${KEY_FILE:-}"
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"

# ---- 打印关键配置（脱敏：不打印 key 本身）----
echo "[retrieval] backend=${SEARCH_BACKEND} port=${RETRIEVAL_PORT} concurrency=${RETRIEVAL_CONCURRENCY}"
echo "[retrieval] cache_db=${CACHE_DB_PATH} workers=${UVICORN_WORKERS}"

# ---- 启动 uvicorn ----
exec uvicorn deepsearch_rl.retrieval.retrieval_server:app \
  --host 0.0.0.0 \
  --port "${RETRIEVAL_PORT}" \
  --workers "${UVICORN_WORKERS}"
