# -*- coding: utf-8 -*-
"""
Judge 服务便捷启动器（vLLM OpenAI 兼容端点）
================================================

以 subprocess 方式拉起一个独立的 vLLM OpenAI 兼容服务，专供奖励函数做
Answer / Evidence 裁判。默认模型 ``Qwen/Qwen3-8B``（也可换 Qwen2.5-7B-Instruct
或其它自选 instruct 模型）。

等价的手动 vllm 命令（vLLM >= 0.18.0 新 CLI）::

    vllm serve Qwen/Qwen3-8B \
        --served-model-name judge \
        --host 0.0.0.0 --port 8001 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.4 \
        --max-model-len 8192 \
        --dtype bfloat16 \
        --enable-prefix-caching

注意：
- 上面用了 ``--served-model-name judge``，因此客户端请求时 ``JUDGE_MODEL`` 应设为
  ``judge``（与 served 名一致）；若省略该参数，则用模型路径本身作为模型名。
- 4090(sm_89) 上 vLLM V1 engine 默认走 FA2 后端，无需额外指定 attention backend。
- 8B 模型 BF16 约 16GB，``--gpu-memory-utilization 0.4`` 在 24GB 卡上留足 KV cache。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Optional


def build_vllm_cmd(args: argparse.Namespace) -> List[str]:
    """根据命令行参数拼装 vllm serve 命令（list 形式，便于 subprocess 调用）。"""
    cmd: List[str] = [
        "vllm", "serve", args.model,
        "--served-model-name", args.served_model_name,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--dtype", args.dtype,
    ]
    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="拉起独立的 vLLM OpenAI 兼容 Judge 服务。",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="裁判模型路径 / HF id（默认 Qwen/Qwen3-8B）")
    parser.add_argument("--served-model-name", default="judge",
                        help="对外暴露的模型名（客户端 JUDGE_MODEL 需与此一致，默认 judge）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enable-prefix-caching", action="store_true", default=True,
                        help="开启 prefix caching（默认开启）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印拼装出的命令，不真正启动")

    args = parser.parse_args(argv)
    cmd = build_vllm_cmd(args)

    print("[judge_server] 启动命令：", " ".join(cmd), flush=True)
    if args.dry_run:
        return 0

    # 直接 exec 等价于把当前进程替换为 vllm；用 subprocess.run 可跨平台、并把
    # vllm 日志透传到当前终端。
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("[judge_server] 收到中断，退出。", flush=True)
        return 0
    except FileNotFoundError:
        print("[judge_server] 找不到 vllm 可执行文件，请先 `pip install vllm>=0.18.0`。",
              file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"[judge_server] vllm 异常退出，code={exc.returncode}", file=sys.stderr)
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
