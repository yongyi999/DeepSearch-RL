# -*- coding: utf-8 -*-
"""
DeepSearch-RL 数据下载与统一中间格式转换
=========================================

本脚本只负责「下载 + 字段归一」，不做任何训练/评测层面的抽样。
产出统一中间 jsonl，字段为：

    id, source, question, gold_answers, type, level, num_hops,
    supporting_titles, split   （MuSiQue 额外带 answerable，供评测过滤）

支持的数据源（--sources）：
    nq        google-research-datasets/nq_open            （单跳，真实 query 分布）
    hotpotqa  hotpotqa/hotpot_qa (config=fullwiki)         （2 跳，bridge/comparison）
    2wiki     voidful/2WikiMultihopQA                     （2 跳 Wikidata 推理）
    musique   dgslibisey/MuSiQue                          （2-4 跳组合推理）
    bamboogle 官方 github json（125 题，仅测试集）          （2 跳对抗题）

后端策略（--backend）：
    auto / modelscope：优先用 modelscope 的 MsDataset 拉取；失败则回退 HF datasets。
    huggingface     ：直接用 HF datasets；若未设置 HF_ENDPOINT，提示并默认指向
                      https://hf-mirror.com 镜像（国内加速）。

注意：本脚本不真正触发大文件下载——在目标机器（AutoDL Ubuntu）上直接运行即可。
已存在的输出文件会跳过（断点续跑）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from hashlib import md5
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# 数据源注册表：HF id / config / ModelScope id / split 映射
# ---------------------------------------------------------------------------
# 说明：ModelScope 上多数社区镜像与 HF id 不完全一致，这里给出「已知」的 MS id；
# 为 None 的源在 auto 模式下会直接尝试 HF（并自动走 hf-mirror）。
SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nq": {
        "hf_id": "google-research-datasets/nq_open",
        "hf_config": None,
        "ms_id": None,
        "splits": ["train", "validation"],
    },
    "hotpotqa": {
        "hf_id": "hotpotqa/hotpot_qa",
        "hf_config": "fullwiki",
        "ms_id": None,
        "splits": ["train", "validation"],
    },
    "2wiki": {
        "hf_id": "voidful/2WikiMultihopQA",
        "hf_config": None,
        "ms_id": None,
        "splits": ["train", "validation"],
    },
    "musique": {
        "hf_id": "dgslibisey/MuSiQue",
        "hf_config": None,
        "ms_id": None,
        "splits": ["train", "validation"],
    },
    "bamboogle": {
        # 官方 json，无 train split，只有 validation
        "url": "https://raw.githubusercontent.com/ofirpress/self-ask/master/data/bamboogle_2hop.json",
        "splits": ["validation"],
    },
}

# HF 国内镜像
HF_MIRROR = "https://hf-mirror.com"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _qid(*parts: Any) -> str:
    """用内容生成稳定 id（避免不同源 id 命名空间冲突）。"""
    s = "||".join(str(p) for p in parts)
    return md5(s.encode("utf-8")).hexdigest()[:16]


def _ensure_list_answers(ans: Any) -> List[str]:
    """把答案字段统一成 list[str]，并去空白、去重保序。"""
    if ans is None:
        return []
    if isinstance(ans, str):
        out = [ans.strip()]
    elif isinstance(ans, (list, tuple)):
        out = [str(a).strip() for a in ans if str(a).strip()]
    else:
        out = [str(ans).strip()]
    # 去重保序
    seen, res = set(), []
    for a in out:
        if a and a not in seen:
            seen.add(a)
            res.append(a)
    return res


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    """流式写 jsonl，返回写入条数。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# 后端加载（惰性导入，避免本地没装 datasets/modelscope 时连 py_compile 都过不了）
# ---------------------------------------------------------------------------
def _load_via_modelscope(ms_id: str, config: Optional[str], split: str):
    """尝试用 ModelScope MsDataset 加载某个 split，失败抛异常由上层回退。"""
    try:
        from modelscope.msdatasets import MsDataset  # 惰性导入
    except Exception as e:  # pragma: no cover - 取决于环境
        raise RuntimeError(f"[modelscope] 未安装或导入失败：{e}")

    kwargs: Dict[str, Any] = {}
    if config:
        kwargs["subset_name"] = config
    ds = MsDataset.load(ms_id, split=split, **kwargs)
    return ds


def _load_via_hf(hf_id: str, config: Optional[str], split: str):
    """用 HuggingFace datasets 加载；自动设置镜像端点。"""
    if not os.environ.get("HF_ENDPOINT"):
        print(f"[hf] 未检测到 HF_ENDPOINT，自动使用镜像 {HF_MIRROR}（可用 export HF_ENDPOINT 覆盖）")
        os.environ["HF_ENDPOINT"] = HF_MIRROR
    try:
        from datasets import load_dataset  # 惰性导入
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "[huggingface] 未安装 datasets：请先 `pip install datasets huggingface_hub`；"
            f"原始导入错误：{e}"
        )

    if config:
        return load_dataset(hf_id, config, split=split)
    return load_dataset(hf_id, split=split)


def load_split(source: str, spec: Dict[str, Any], split: str, backend: str):
    """根据 backend 选择加载方式，返回可迭代的原始行。"""
    if source == "bamboogle":
        return _load_bamboogle(spec["url"])

    hf_id = spec["hf_id"]
    config = spec.get("hf_config")
    ms_id = spec.get("ms_id")

    ms_tried = False
    if backend in ("auto", "modelscope") and ms_id:
        ms_tried = True
        try:
            print(f"[{source}] 尝试 modelscope 加载 {ms_id} ({split}) ...")
            return _load_via_modelscope(ms_id, config, split)
        except Exception as e:
            print(f"[{source}] modelscope 失败，回退 HF：{e}")

    # huggingface 或 auto 回退
    try:
        return _load_via_hf(hf_id, config, split)
    except Exception as e:
        hint = ""
        if backend == "modelscope" and not ms_tried:
            hint = "（该源未配置 ModelScope id，可改用 --backend auto 或 huggingface）"
        raise RuntimeError(
            f"[{source}] 加载 {hf_id} ({split}) 失败{hint}。\n"
            f"  常见原因：\n"
            f"    1) 未安装依赖：pip install -U datasets huggingface_hub modelscope\n"
            f"    2) 网络不通：国内请设置 export HF_ENDPOINT=https://hf-mirror.com\n"
            f"    3) 仓库需要 trust_remote_code：本脚本未开启，若报错请手动调整。\n"
            f"  原始错误：{e}"
        )


def _load_bamboogle(url: str) -> List[Dict[str, Any]]:
    """Bamboogle 是官方 GitHub 上的一个静态 json（list[dict]），直接 HTTP 拉取。"""
    print(f"[bamboogle] 下载官方 json：{url}")
    req = urllib.request.Request(url, headers={"User-Agent": "deepsearch-rl/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise RuntimeError(f"[bamboogle] 官方 json 顶层应为 list，实际为 {type(data)}")
    return data


# ---------------------------------------------------------------------------
# 各源的字段归一函数：原始行 -> 统一中间 dict
# ---------------------------------------------------------------------------
def normalize_nq(row: Dict[str, Any]) -> Dict[str, Any]:
    # nq_open: question(str), answer(list[str])
    question = row["question"]
    gold = _ensure_list_answers(row.get("answer"))
    return {
        "id": f"nq_{_qid(question, gold)}",
        "source": "nq",
        "question": question,
        "gold_answers": gold,
        "type": "single-hop",
        "level": "easy",
        "num_hops": 1,
        "supporting_titles": [],
    }


def normalize_hotpotqa(row: Dict[str, Any]) -> Dict[str, Any]:
    # hotpotqa fullwiki: id, question, answer(str), type, level, supporting_facts{title:[...], sent_id:[...]}
    question = row["question"]
    gold = _ensure_list_answers(row.get("answer"))
    sf = row.get("supporting_facts") or {}
    titles: List[str] = []
    if isinstance(sf, dict):
        titles = [t for t in (sf.get("title") or []) if t]
    elif isinstance(sf, list):
        # 兼容 [[title, sent_id], ...] 形式
        for item in sf:
            if isinstance(item, (list, tuple)) and item:
                titles.append(str(item[0]))
    return {
        "id": f"hotpot_{row.get('id') or _qid(question)}",
        "source": "hotpotqa",
        "question": question,
        "gold_answers": gold,
        "type": row.get("type", "unknown"),
        "level": row.get("level", "unknown"),
        "num_hops": 2,
        "supporting_titles": titles,
    }


def normalize_2wiki(row: Dict[str, Any]) -> Dict[str, Any]:
    # 2Wiki: question_id, question, answer(str), type, evidences(list[dict{title,...}]),
    #        supporting_facts / supported_evidence, context{title:[...]}
    question = row["question"]
    gold = _ensure_list_answers(row.get("answer"))
    titles: List[str] = []

    # 1) evidences 是 list[dict]，含 title
    ev = row.get("evidences")
    if isinstance(ev, list):
        for e in ev:
            if isinstance(e, dict) and e.get("title"):
                titles.append(str(e["title"]))
    # 2) supporting_facts 可能是 [[title, sent_id], ...]
    if not titles:
        sf = row.get("supporting_facts")
        if isinstance(sf, list):
            for item in sf:
                if isinstance(item, (list, tuple)) and item:
                    titles.append(str(item[0]))
                elif isinstance(item, dict) and item.get("title"):
                    titles.append(str(item["title"]))
    # 3) supported_evidence + context.title 兜底
    if not titles:
        sup = row.get("supported_evidence") or {}
        ctx = row.get("context") or {}
        if isinstance(sup, dict) and isinstance(ctx, dict):
            ctx_titles = ctx.get("title") or []
            ids = sup.get("evidence_id") or []
            for i in ids:
                try:
                    titles.append(str(ctx_titles[int(i)]))
                except Exception:
                    pass

    # 去重保序
    seen, uniq = set(), []
    for t in titles:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)

    qid = row.get("question_id") or row.get("id") or _qid(question)
    return {
        "id": f"2wiki_{qid}",
        "source": "2wiki",
        "question": question,
        "gold_answers": gold,
        "type": row.get("type", "unknown"),
        "level": row.get("level", "hard"),
        "num_hops": 2,
        "supporting_titles": uniq,
    }


def normalize_musique(row: Dict[str, Any]) -> Dict[str, Any]:
    # MuSiQue: id, paragraphs(list[dict{title, is_supporting, paragraph_text}]),
    #          question, answer(str), answer_aliases(list), answerable(bool),
    #          question_decomposition(list[dict])
    question = row["question"]
    gold = _ensure_list_answers([row.get("answer")] + list(row.get("answer_aliases") or []))

    titles: List[str] = []
    for p in row.get("paragraphs") or []:
        if isinstance(p, dict) and p.get("is_supporting") and p.get("title"):
            titles.append(str(p["title"]))

    # num_hops：用 question_decomposition 长度估计，兜底 2
    dec = row.get("question_decomposition") or []
    num_hops = len(dec) if isinstance(dec, list) and dec else 2

    rec = {
        "id": f"musique_{row.get('id') or _qid(question)}",
        "source": "musique",
        "question": question,
        "gold_answers": gold,
        "type": "compositional",
        "level": "hard",
        "num_hops": num_hops,
        "supporting_titles": titles,
    }
    # 评测期需要按 answerable 过滤，挂在中间记录上
    if "answerable" in row:
        rec["answerable"] = bool(row["answerable"])
    return rec


def normalize_bamboogle(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    # 官方 json: question(str), golden_answers(list[str])
    question = row["question"]
    gold = _ensure_list_answers(row.get("golden_answers") or row.get("answer"))
    return {
        "id": f"bamboogle_{idx:04d}",
        "source": "bamboogle",
        "question": question,
        "gold_answers": gold,
        "type": "bridge",
        "level": "hard",
        "num_hops": 2,
        "supporting_titles": [],
    }


NORMALIZERS = {
    "nq": normalize_nq,
    "hotpotqa": normalize_hotpotqa,
    "2wiki": normalize_2wiki,
    "musique": normalize_musique,
    "bamboogle": None,  # 单独处理
}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process_source(source: str, out_dir: str, backend: str, nq_limit: Optional[int]) -> None:
    spec = SOURCE_REGISTRY[source]
    os.makedirs(out_dir, exist_ok=True)

    for split in spec["splits"]:
        out_path = os.path.join(out_dir, f"{source}_{split}.jsonl")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} 已存在，跳过。")
            continue

        t0 = time.time()
        rows = load_split(source, spec, split, backend)

        if source == "bamboogle":
            norm_rows = [normalize_bamboogle(r, i) for i, r in enumerate(rows)]
        else:
            norm_fn = NORMALIZERS[source]
            norm_rows = []
            for i, r in enumerate(rows):
                # NQ-open train：下载期可选硬截断（--nq_limit，0 表示不截断）。
                # 注意：训练配比里「可复现的 NQ 降采样」由 prepare_train.py 按 seed=42 完成；
                # 这里的截断只是为了省磁盘，若想要全量 87k 再 seed-42 抽样，请传 --nq_limit 0。
                if (
                    source == "nq"
                    and split == "train"
                    and nq_limit
                    and i >= nq_limit
                ):
                    break
                norm_rows.append(norm_fn(dict(r)))

        n = _write_jsonl(out_path, norm_rows)
        print(f"[ok] {source}/{split}: 写出 {n} 条 -> {out_path}（用时 {time.time()-t0:.1f}s）")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="下载多跳 QA 数据集并归一为统一中间 jsonl")
    parser.add_argument("--out_dir", default="data/raw", help="输出目录（默认 data/raw）")
    parser.add_argument(
        "--sources",
        default="nq,hotpotqa,2wiki,musique,bamboogle",
        help="逗号分隔的源列表（默认全部）",
    )
    parser.add_argument(
        "--nq_limit",
        type=int,
        default=30000,
        help="NQ-open train 下载期硬截断上限（默认 30000；传 0 表示拉全量 ~87925）。"
        "训练配比的可复现降采样在 prepare_train.py 里按 seed=42 完成。",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "modelscope", "huggingface"],
        default="auto",
        help="下载后端（默认 auto：优先 modelscope，失败回退 HF datasets）",
    )
    args = parser.parse_args(argv)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    bad = [s for s in sources if s not in SOURCE_REGISTRY]
    if bad:
        print(f"[error] 未知数据源：{bad}；可选：{list(SOURCE_REGISTRY)}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    for src in sources:
        try:
            process_source(src, args.out_dir, args.backend, args.nq_limit)
        except Exception as e:
            print(f"[error] 源 {src} 处理失败：\n{e}", file=sys.stderr)
            # 单源失败不影响其他源，但返回非零
            return 1

    print("\n全部完成。中间文件位于：", os.path.abspath(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
