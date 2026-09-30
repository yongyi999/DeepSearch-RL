# -*- coding: utf-8 -*-
"""
DeepSearch-RL 训练集预处理：统一中间 jsonl -> veRL parquet
==========================================================

读取 download_data.py 产出的 data/raw/*.jsonl，按 SPEC 4.1 生成 veRL 可直接消费的
parquet。每行字段严格为：

    data_source   : "nq" | "hotpotqa" | "2wiki" | "musique"
    prompt        : list[dict]，[{"role":"system",...}, {"role":"user",...}]
                    （veRL 要求 prompt 为 chat 列表，而非单条字符串）
    ability       : "multi-hop-search"
    agent_name    : "deepsearch_agent"
    reward_model  : {"style":"rule", "ground_truth":{"target":[gold,...]}}
    extra_info    : {"split":"train|val", "index":int, "num_hops":int}

训练混合口径（SPEC 4.1）：
    - NQ-open train 降采样到 --nq_limit（默认 30000，固定 seed 42）
    - HotpotQA / 2Wiki / MuSiQue 的 train 全量
    - Bamboogle 不进训练集
    - 从每个 train 源固定抽 200 条做训练期 val（seed 42）

本模块顶层只 import 标准库 + deepsearch_rl.protocol（后者无重依赖），
pyarrow / 文件 IO 在函数内惰性导入，保证 record_to_parquet_row 可离线单测。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

# 让脚本在任意 cwd 下都能 import 工程根目录的 deepsearch_rl
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deepsearch_rl.protocol import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

# 训练集只取这 4 个源（Bamboogle 仅评测）
TRAIN_SOURCES = ["nq", "hotpotqa", "2wiki", "musique"]
# 每个源抽多少条做训练期 val
VAL_PER_SOURCE = 200


# ---------------------------------------------------------------------------
# 纯函数：单条统一中间记录 -> veRL parquet 行（可离线单测，不触网）
# ---------------------------------------------------------------------------
def record_to_parquet_row(record: Dict[str, Any], *, split: str, index: int) -> Dict[str, Any]:
    """把一条统一中间 jsonl 记录转成 veRL parquet 的一行 dict。

    Args:
        record: download_data 产出的统一记录（至少含 source/question/gold_answers/num_hops）。
        split:  "train" 或 "val"。
        index:  该条在当前 shard 内的序号（写入 extra_info.index）。

    Returns:
        一个可直接被 pyarrow.Table.from_pylist 消费的 dict。
    """
    question = record["question"]
    gold_answers = list(record.get("gold_answers") or [])

    prompt: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question)},
    ]

    return {
        "data_source": record.get("source", "unknown"),
        "prompt": prompt,
        "ability": "multi-hop-search",
        "agent_name": "deepsearch_agent",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": gold_answers},
        },
        "extra_info": {
            "split": split,
            "index": int(index),
            "num_hops": int(record.get("num_hops", 0) or 0),
        },
    }


# ---------------------------------------------------------------------------
# 读 raw jsonl
# ---------------------------------------------------------------------------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_train_sources(raw_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """读取各源 train 文件，返回 {source: rows}。缺文件则报错。"""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for src in TRAIN_SOURCES:
        path = os.path.join(raw_dir, f"{src}_train.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到训练文件：{path}\n"
                f"请先运行 download_data.py（--sources {src}）生成。"
            )
        out[src] = _read_jsonl(path)
        print(f"[load] {src}_train.jsonl : {len(out[src])} 条")
    return out


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------
def _downsample(rows: List[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    """固定 seed 随机抽 limit 条；limit<=0 或 >= 总数时原样返回。"""
    if limit and limit < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, limit)
    return rows


# ---------------------------------------------------------------------------
# 写 parquet（pyarrow 惰性导入）
# ---------------------------------------------------------------------------
def _write_parquet(rows: List[Dict[str, Any]], path: str) -> None:
    import pyarrow as pa  # 惰性导入
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_dataset(
    raw_dir: str,
    out_dir: str,
    nq_limit: int = 30000,
    seed: int = 42,
) -> Dict[str, int]:
    """读取 raw，构建 train/val parquet，返回各文件条数统计。"""
    data = _load_train_sources(raw_dir)

    # 1) NQ 降采样
    data["nq"] = _downsample(data["nq"], nq_limit, seed)
    print(f"[sample] nq 降采样到 {len(data['nq'])}（seed={seed}）")

    # 2) 从每个 train 源抽 val（在 NQ 降采样之后抽，保证 val 不越界）
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    for src in TRAIN_SOURCES:
        rows = data[src]
        val_pool = _downsample(rows, VAL_PER_SOURCE, seed)
        val_ids = {r["id"] for r in val_pool}
        # val 是 train 的子集；train 里保留其余 + val（veRL 训练时用全部 train）
        train_rows.extend(rows)
        val_rows.extend(val_pool)
        print(f"[split] {src}: train={len(rows)} val={len(val_pool)}")

    # 3) 转 parquet 行
    train_out = [
        record_to_parquet_row(r, split="train", index=i)
        for i, r in enumerate(train_rows)
    ]
    val_out = [
        record_to_parquet_row(r, split="val", index=i)
        for i, r in enumerate(val_rows)
    ]

    # 4) 写出（按源分 shard，便于排查）
    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")

    # 重新按源聚合写 shard
    stats: Dict[str, int] = {}
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for src in TRAIN_SOURCES:
        src_train = [r for r in train_out if r["data_source"] == src]
        src_val = [r for r in val_out if r["data_source"] == src]
        tp = os.path.join(train_dir, f"{src}.parquet")
        vp = os.path.join(val_dir, f"{src}.parquet")
        _write_parquet(src_train, tp)
        _write_parquet(src_val, vp)
        stats[f"train/{src}"] = len(src_train)
        stats[f"val/{src}"] = len(src_val)
        print(f"[write] {tp} ({len(src_train)} 行) ; {vp} ({len(src_val)} 行)")

    stats["train_total"] = len(train_out)
    stats["val_total"] = len(val_out)
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="统一中间 jsonl -> veRL 训练 parquet")
    parser.add_argument("--raw_dir", default="data/raw", help="download_data 产出目录（默认 data/raw）")
    parser.add_argument("--out_dir", default="data/processed", help="parquet 输出根目录（默认 data/processed）")
    parser.add_argument("--nq_limit", type=int, default=30000, help="NQ-open train 降采样上限（默认 30000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args(argv)

    stats = build_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        nq_limit=args.nq_limit,
        seed=args.seed,
    )
    print("\n=== 训练/val 条数统计 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
