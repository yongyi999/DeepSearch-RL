# -*- coding: utf-8 -*-
"""
DeepSearch-RL 500 题冻结评测集构建
==================================

严格按 SPEC 4.2 / data_deps_research 定稿配额（seed=42）：

    HotpotQA hard (不足补 comparison)  150
    2WikiMultihopQA validation        125
    MuSiQue validation (answerable)  100
    Bamboogle（全集，仅 125）         125
    ---------------------------------------
    合计                             500

只用各源 validation/dev split（绝不与 train 重叠），一次性抽样后冻结为
data/eval_hard_500.jsonl。输出字段：
    id, source, question, gold_answers, type, level, num_hops, supporting_titles, split
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

# 配额（合计必须为 500）
QUOTA_HOTPOTQA = 150
QUOTA_2WIKI = 125
QUOTA_MUSIQUE = 100
QUOTA_BAMBOOGLE = 125


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample(pool: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    """从 pool 里随机抽 k 条；pool 不足时全取。"""
    if k <= 0 or not pool:
        return []
    if len(pool) <= k:
        return list(pool)
    return rng.sample(pool, k)


def _select_hotpotqa(rows: List[Dict[str, Any]], quota: int, rng: random.Random) -> List[Dict[str, Any]]:
    """先抽 level==hard 的；不足 quota 时，用 type==comparison 且未被抽的补齐。"""
    hard = [r for r in rows if r.get("level") == "hard"]
    picked = _sample(hard, quota, rng)
    if len(picked) < quota:
        picked_ids = {r["id"] for r in picked}
        comparison = [
            r for r in rows if r.get("type") == "comparison" and r["id"] not in picked_ids
        ]
        need = quota - len(picked)
        picked = picked + _sample(comparison, need, rng)
    return picked


def build_eval(raw_dir: str, out_path: str, seed: int = 42) -> Dict[str, int]:
    """读取各源 validation，按配额抽样，写出 eval_hard_500.jsonl。返回各源实际条数。"""
    rng = random.Random(seed)

    def _load(name: str) -> List[Dict[str, Any]]:
        p = os.path.join(raw_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"找不到评测源文件：{p}\n请先运行 download_data.py 生成 validation 中间文件。"
            )
        return _read_jsonl(p)

    # HotpotQA validation：hard 优先，不足补 comparison
    hp_rows = _load("hotpotqa_validation.jsonl")
    hp_picked = _select_hotpotqa(hp_rows, QUOTA_HOTPOTQA, rng)

    # 2Wiki validation：全类型抽样
    wiki_rows = _load("2wiki_validation.jsonl")
    wiki_picked = _sample(wiki_rows, QUOTA_2WIKI, rng)

    # MuSiQue validation：仅 answerable
    mus_rows = _load("musique_validation.jsonl")
    mus_pool = [r for r in mus_rows if r.get("answerable", True)]
    mus_picked = _sample(mus_pool, QUOTA_MUSIQUE, rng)

    # Bamboogle：全集（恰好 125）
    bam_rows = _load("bamboogle_validation.jsonl")
    bam_picked = list(bam_rows)

    picked = hp_picked + wiki_picked + mus_picked + bam_picked

    # 字段裁剪：只保留 SPEC 4.2 规定的 9 个字段
    KEEP = {
        "id", "source", "question", "gold_answers",
        "type", "level", "num_hops", "supporting_titles", "split",
    }
    out_rows: List[Dict[str, Any]] = []
    for r in picked:
        rec = {k: r.get(k) for k in KEEP}
        # split 统一标 validation（这些都来自 dev/validation）
        rec["split"] = "validation"
        out_rows.append(rec)

    # 唯一性校验
    ids = [r["id"] for r in out_rows]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"[error] id 有 {dup} 个重复，请检查数据源。")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "hotpotqa(hard/comparison)": len(hp_picked),
        "2wiki": len(wiki_picked),
        "musique(answerable)": len(mus_picked),
        "bamboogle": len(bam_picked),
        "total": len(out_rows),
    }
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="构建 500 题冻结 hard multi-hop 评测集")
    parser.add_argument("--raw_dir", default="data/raw", help="download_data 产出目录（默认 data/raw）")
    parser.add_argument("--out", default="data/eval_hard_500.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args(argv)

    stats = build_eval(args.raw_dir, args.out, args.seed)

    print("=== 500 题评测集实际配额 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats["total"] != 500:
        print(f"[warn] 合计 {stats['total']} != 500；可能是某源 validation 不足，请检查。", file=sys.stderr)
        return 1
    print(f"\n已写出：{os.path.abspath(args.out)}（固定 seed={args.seed}，冻结后请勿重抽）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
