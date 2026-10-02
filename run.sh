#!/usr/bin/env bash
# =============================================================================
# DeepSearch-RL 一键编排（工程根目录）
#
# 用法：
#   bash run.sh retrieval     # 终端 A：起检索服务（FastAPI，:8000）
#   bash run.sh judge          # 终端 B：起远程 Judge（vLLM openai 协议，:8001）
#   bash run.sh train          # 终端 C：起 veRL 训练（6×RTX 4090）
#   bash run.sh eval           # 跑 500 题冻结评测（eval_hard_500.jsonl）
#   bash run.sh all            # 打印「分别开 3 个终端」的标准启动说明（默认演示）
#   bash run.sh install        # 调用 scripts/install_autodl.sh 装环境
#   bash run.sh model          # 调用 scripts/download_model.sh 下模型
#
# 注意：retrieval / judge / train 是三个**独立常驻进程**，必须分别在不同终端里
#       执行（它们之间通过 HTTP / 环境变量解耦），不能在同一个 bash 里串起来跑。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

cmd="${1:-all}"

case "$cmd" in
    retrieval)
        echo "[run.sh] 启动检索服务（终端 A）..."
        exec bash scripts/start_retrieval.sh
        ;;
    judge)
        echo "[run.sh] 启动 Judge 服务（终端 B）..."
        exec bash scripts/start_judge.sh
        ;;
    train)
        echo "[run.sh] 启动训练（终端 C）..."
        exec bash scripts/train.sh "${@:2}"
        ;;
    eval)
        echo "[run.sh] 运行 500 题评测..."
        exec python -m deepsearch_rl.eval.evaluate "${@:2}"
        ;;
    install)
        echo "[run.sh] 安装 AutoDL 环境..."
        exec bash scripts/install_autodl.sh
        ;;
    model)
        echo "[run.sh] 下载 Qwen3-8B 权重..."
        exec bash scripts/download_model.sh
        ;;
    all|--help|-h|help)
        cat <<'EOF'
DeepSearch-RL 标准启动流程（请分别在 3 个终端里执行，不要合到一起）：

  [前置]
    export MODEL_PATH=$HOME/models/Qwen3-8B
    export SWANLAB_API_KEY=xxxx          # 或 export SWANLAB_MODE=offline

  [终端 A —— 检索服务（常驻）]
    bash run.sh retrieval
    # 监听 127.0.0.1:8000，提供 /retrieve /open /health

  [终端 B —— 远程 Judge（常驻）]
    bash run.sh judge
    # 监听 127.0.0.1:8001/v1，供 reward 函数判答案/证据充分度

  [终端 C —— 训练]
    bash run.sh train
    # 等价于 bash scripts/train.sh，后面可追加 Hydra 覆盖，例如：
    #   bash run.sh train data.train_batch_size=128 actor_rollout_ref.rollout.n=5

  [训练结束后]
    bash run.sh eval                     # 在冻结 500 题上评测

子命令：retrieval | judge | train | eval | install | model | all
EOF
        ;;
    *)
        echo "[错误] 未知子命令：$cmd"
        echo "       可选：retrieval | judge | train | eval | install | model | all"
        exit 2
        ;;
esac
