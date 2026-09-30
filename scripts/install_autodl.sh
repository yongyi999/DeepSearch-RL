#!/usr/bin/env bash
# =============================================================================
# DeepSearch-RL AutoDL 环境一键安装（Ubuntu + 4×RTX 5090 32GB / sm_120）
#
# 目标：Python 3.12 + CUDA 12.8 + PyTorch 2.8.0+cu128 + verl v0.6.0 + sglang(≤0.5.19)
#
# 用法（在 AutoDL 实例内，工程根目录执行）：
#   bash scripts/install_autodl.sh
#
# 设计：
# - set -e：任一步失败立即停；
# - 可重复执行：已存在的目录/包会跳过或覆盖安装，不会把环境搞坏；
# - 不真正下载模型/数据，只搭训练环境。
# =============================================================================
set -euo pipefail

# 切到工程根目录（本脚本位于 scripts/ 下）
cd "$(dirname "$0")/.."

echo "======================================================================"
echo "[1/7] 基础环境自检（python / pip / nvcc）"
echo "======================================================================"
# AutoDL 镜像一般已带 conda python；这里只做提示，不强制建环境
PY=${PYTHON:-python}
echo "[info] 使用解释器： $($PY --version 2>&1)"
echo "[info] pip：       $($PY -m pip --version 2>&1)"

# AutoDL 常见：CUDA 12.8 已预装；若没有 nvcc 提示一下（不强制）
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[提示] 未检测到 nvcc（仅编译 flash-attn 等需要；本流程用 liger-kernel 免编译，可忽略）"
fi

# ---------------------------------------------------------------------------
# [2/7] 安装 PyTorch 2.8.0 + cu128（5090 sm_120 必须 cu128 车道）
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[2/7] 安装 torch==2.8.0 cu128"
echo "======================================================================"
$PY -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# [3/7] 安装通用 python 依赖（对齐 requirements.txt 的版本口径）
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[3/7] 安装通用依赖（modelscope / datasets / ray / swanlab 等）"
echo "======================================================================"
# 若工程里有 requirements.txt 直接安装；否则按下面的关键版本号兜底
if [[ -f requirements.txt ]]; then
    $PY -m pip install -r requirements.txt
else
    # 与 requirements.txt 对齐的最小集合（防 requirements.txt 缺失时脚本仍可跑）
    $PY -m pip install \
        "modelscope==1.23.1" \
        "datasets>=2.20.0" "huggingface_hub>=0.25.0" "pyarrow>=15.0.0" \
        "ray[data,train,tune,serve]>=2.45.0,<2.49" \
        "hydra-core>=1.3.2" \
        "transformers>=4.51.0" "accelerate>=0.34.0" "peft>=0.12.0" \
        "numpy<2.3" pandas dill pylatexenc tiktoken codetiming \
        "swanlab==0.9.0" \
        "openai>=1.40.0" "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" \
        "tenacity>=2.3.0" "aiohttp>=3.10.0" "requests>=2.32.0" \
        "trafilatura>=1.12.0" "beautifulsoup4>=4.12.0" "lxml>=5.3.0"
fi

# ---------------------------------------------------------------------------
# [4/7] clone verl 并 checkout v0.6.0（必须 pin，main 已迁 CUDA13/torch2.14）
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[4/7] 安装 verl v0.6.0（源码 editable，含 sglang extra）"
echo "======================================================================"
VERL_DIR=${VERL_DIR:-$HOME/verl}
if [[ ! -d "$VERL_DIR" ]]; then
    git clone https://github.com/volcengine/verl.git "$VERL_DIR"
else
    echo "[info] 已存在 $VERL_DIR，跳过 clone"
fi
cd "$VERL_DIR"
git fetch --tags --force
git checkout v0.6.0
# 装 verl 本体 + sglang extra（会顺带 pin sglang 到 0.5.x）
$PY -m pip install -e ".[sglang]"
cd - >/dev/null

# ---------------------------------------------------------------------------
# [5/7] sglang 版本二次固定 + flashinfer (cu128)
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[5/7] 固定 sglang<=0.5.19 并安装 flashinfer-cu128"
echo "======================================================================"
# verl 的 requirements_sglang.txt 已 pin；这里再显式固定一次，防止 pip 自动升新
$PY -m pip install "sglang>=0.4.6.post1,<0.5.20"
# flashinfer 必须 cu128 车道（5090 sm_120）；用 sglang 推荐的 find-links
$PY -m pip install \
    "flashinfer_python" \
    --find-links https://flashinfer.ai/whl/cu128/torch2.8/flashinfer-python

# ---------------------------------------------------------------------------
# [6/7] liger-kernel（免编译的 kernel，5090 上替代 flash-attn）
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[6/7] 安装 liger-kernel"
echo "======================================================================"
$PY -m pip install "liger-kernel>=0.8.2"

# ---------------------------------------------------------------------------
# [7/7] 自检
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[7/7] 安装完成，自检关键 import"
echo "======================================================================"
$PY - <<'PY'
import importlib, sys
mods = ["torch", "transformers", "ray", "swanlab", "omegaconf", "hydra",
        "verl", "sglang", "liger_kernel", "modelscope", "datasets"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  [ok] {m:14s} {v}")
    except Exception as e:
        print(f"  [FAIL] {m:14s} {type(e).__name__}: {e}")
print("python:", sys.version.split()[0])
PY

echo "======================================================================"
echo "下一步："
echo "  1) bash scripts/download_model.sh   # 下 Qwen3-8B"
echo "  2) export MODEL_PATH=\$HOME/models/Qwen3-8B"
echo "  3) bash scripts/start_retrieval.sh  # 终端 A：检索服务"
echo "  4) bash scripts/start_judge.sh       # 终端 B：Judge 服务"
echo "  5) bash scripts/train.sh             # 终端 C：训练"
echo "======================================================================"
