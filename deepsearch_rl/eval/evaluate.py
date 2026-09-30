# -*- coding: utf-8 -*-
"""
500 题冻结评测入口
====================

对齐 ENGINEERING_SPEC 4.2 / 第 7 节。流程：

a) 加载 ``data/eval_hard_500.jsonl``（字段 question/gold_answers/source/num_hops/
   supporting_titles …）。
b) 用 ``tool_factory`` 构建真实 search/open 工具（含缓存 / key 轮换 / 重试），
   用 :class:`OpenAICompatAgent` 作为 agent（base_url 指向被评模型的 SGLang/vLLM
   OpenAI 端点，注入 search/open 工具，max_turns 控制检索深度）。
c) ``asyncio`` 信号量并发对每题 ``agent.run(question)``，得到 ``StandaloneResult``。
d) 逐题打分：
   - answer_correct：EM 或 best_f1>=f1_threshold 优先命中；否则 Answer Judge。
   - evidence_sufficiency：对 full_text 用 ``TrajectoryAnalyzer`` 取 evidence_text，
     再 Evidence Judge，score>=evidence_threshold 判充分。
   - correct_and_sufficient = 二者同时成立。
   - duplicate_rate = num_duplicate / max(num_tool_calls, 1)。
   - num_search。
e) 汇总 5 个核心指标（与简历口径一致），并按 source 分组。
f) 写 JSON（配置 + 总体 + 分组 + 耗时），控制台打印 Markdown 表；--save_trajectories
   时把每题完整轨迹写 jsonl。

降级约定（SPEC 3.7）：
- Judge 不可达：答案退回 EM/F1 规则，证据退回 supporting-title 命中代理，不抛异常。
- 被评模型端点不可达：捕获连接异常，给清晰中文提示（如何用 SGLang 起模型），
  该题记为 answer=None，不中断整体评测。

用法::

    python -m deepsearch_rl.eval.evaluate \
        --model_endpoint http://127.0.0.1:30000/v1 \
        --model_name Qwen/Qwen3-8B \
        --judge_base_url http://127.0.0.1:8001/v1 \
        --judge_model judge

或用 ``--compare baseline.json trained.json`` 只做两份结果对比。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ----------------------------------------------------------------------------
# 纯逻辑依赖（无重三方库，顶层导入安全）
# ----------------------------------------------------------------------------
from ..agent.standalone_agent import StandaloneResult
from ..agent.trajectory import TrajectoryAnalyzer
from ..rewards.answer_metrics import best_f1, em_match

# JudgeClient 本身顶层 import openai / tenacity（项目已依赖）。
# 若环境连 openai 都没装，评测主流程也无法连模型，这里给出友好提示。
try:  # pragma: no cover - 取决于运行环境
    from ..judge.judge_client import JudgeClient

    _HAS_JUDGE_CLIENT = True
except Exception:  # pragma: no cover
    JudgeClient = Any  # type: ignore
    _HAS_JUDGE_CLIENT = False

# 五个核心指标的固定展示顺序（与简历口径一致）
METRIC_ORDER = [
    ("accuracy", "Accuracy", "pct"),
    ("evidence_sufficiency", "Evidence Suff.", "pct"),
    ("correct_and_sufficient", "Correct & Suff.", "pct"),
    ("duplicate_call_rate", "Duplicate Call Rate", "pct"),
    ("avg_search_per_query", "Avg Search/query", "f1"),
]

# 默认按 source 分组的固定顺序
SOURCE_ORDER = ["hotpotqa", "2wiki", "musique", "bamboogle"]


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class EvalThresholds:
    """打分阈值（集中一处，便于对齐 SPEC 与 yaml）。"""

    f1_correct: float = 0.9          # best_f1 >= 此值视为答案正确（跳过 Judge）
    evidence_sufficient: float = 0.6  # Evidence Judge score >= 此值视为证据充分


@dataclass
class PerItemScore:
    """单题评测结果。"""

    id: str
    source: str
    question: str
    answer: Optional[str]
    gold_answers: List[str]
    num_search: int
    num_open: int
    num_duplicate: int
    num_tool_calls: int
    duplicate_rate: float
    answer_correct: bool
    evidence_sufficient: bool
    correct_and_sufficient: bool
    answer_method: str = ""        # 命中来源：em / f1 / judge / judge_unreachable
    evidence_method: str = ""       # judge / title_hit / judge_unreachable
    error: Optional[str] = None    # 该题若执行失败，记录原因

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "num_search": self.num_search,
            "num_open": self.num_open,
            "num_duplicate": self.num_duplicate,
            "num_tool_calls": self.num_tool_calls,
            "duplicate_rate": round(self.duplicate_rate, 4),
            "answer_correct": self.answer_correct,
            "evidence_sufficient": self.evidence_sufficient,
            "correct_and_sufficient": self.correct_and_sufficient,
            "answer_method": self.answer_method,
            "evidence_method": self.evidence_method,
            "error": self.error,
        }


# ============================================================================
# 数据加载
# ============================================================================
def load_eval_rows(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """读取冻结评测 jsonl；逐行容错，空行跳过。limit 仅取前 N 题（调试用）。"""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


# ============================================================================
# 单题打分（纯异步，便于离线注入 FakeAgent 单测）
# ============================================================================
async def score_one(
    row: Dict[str, Any],
    result: StandaloneResult,
    judge: Optional[JudgeClient],
    judge_available: bool,
    thresholds: EvalThresholds,
) -> PerItemScore:
    """对单题的 StandaloneResult 计算全部评测指标。

    Args:
        row: 评测题（含 gold_answers / supporting_titles / source）。
        result: agent.run(question) 的产出。
        judge: Judge 客户端；为 None 或 judge_available=False 时走规则代理。
        judge_available: 启动时探测到 Judge 是否真的可用。
        thresholds: 打分阈值。
    """
    qid = str(row.get("id", ""))
    source = str(row.get("source", "unknown"))
    question = row.get("question", "")
    gold_answers = list(row.get("gold_answers") or [])
    answer = result.answer

    # --- 工具调用统计（以 StandaloneResult 实际执行数为准） ---
    num_search = int(result.num_search)
    num_open = int(result.num_open)
    num_tool_calls = num_search + num_open
    # num_duplicate 由 StandaloneResult 内部经 TrajectoryAnalyzer 算出；兜底再用一次
    num_duplicate = int(result.num_duplicate)
    duplicate_rate = num_duplicate / max(num_tool_calls, 1)

    # --- 答案正确性：EM / 高 F1 优先，否则 Judge ---
    em = em_match(answer, gold_answers)
    f1 = best_f1(answer, gold_answers)
    answer_correct = False
    answer_method = ""
    if em:
        answer_correct = True
        answer_method = "em"
    elif f1 >= thresholds.f1_correct:
        answer_correct = True
        answer_method = "f1"
    elif judge is not None and judge_available:
        verdict = await judge.judge_answer(question, answer or "", gold_answers)
        answer_correct = bool(verdict.correct)
        answer_method = "judge"
    else:
        # Judge 不可达：规则代理即 EM/F1（上面已算，未命中即为 False）
        answer_method = "rule_only(judge_unreachable)"

    # --- 证据充分度：对 full_text 取 evidence_text，再 Evidence Judge ---
    analyzer = TrajectoryAnalyzer.from_solution(result.full_text or "")
    evidence_text = analyzer.evidence_text
    evidence_sufficient = False
    evidence_method = ""
    if judge is not None and judge_available:
        ev = await judge.judge_evidence(question, answer or "", evidence_text, gold_answers)
        # JudgeClient 内部已按 score>=0.6 兜底；这里再卡一次阈值保证口径一致
        evidence_sufficient = bool(ev.sufficient) and ev.score >= thresholds.evidence_sufficient
        evidence_method = "judge"
    else:
        # Judge 不可达：supporting-title 命中代理——任一 golden 标题出现在证据里即充分
        titles = row.get("supporting_titles") or []
        hit = False
        if titles and evidence_text:
            low_ev = evidence_text.lower()
            hit = any(str(t).strip().lower() in low_ev for t in titles if str(t).strip())
        evidence_sufficient = hit
        evidence_method = "title_hit(judge_unreachable)"

    return PerItemScore(
        id=qid,
        source=source,
        question=question,
        answer=answer,
        gold_answers=gold_answers,
        num_search=num_search,
        num_open=num_open,
        num_duplicate=num_duplicate,
        num_tool_calls=num_tool_calls,
        duplicate_rate=duplicate_rate,
        answer_correct=answer_correct,
        evidence_sufficient=evidence_sufficient,
        correct_and_sufficient=answer_correct and evidence_sufficient,
        answer_method=answer_method,
        evidence_method=evidence_method,
    )


# ============================================================================
# 汇总
# ============================================================================
def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_metrics(items: List[PerItemScore]) -> Dict[str, Any]:
    """总体 + 按 source 分组汇总 5 个核心指标。"""

    def _block(sub: List[PerItemScore]) -> Dict[str, Any]:
        n = len(sub)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "accuracy": _mean([float(i.answer_correct) for i in sub]),
            "evidence_sufficiency": _mean([float(i.evidence_sufficient) for i in sub]),
            "correct_and_sufficient": _mean([float(i.correct_and_sufficient) for i in sub]),
            "duplicate_call_rate": _mean([i.duplicate_rate for i in sub]),
            "avg_search_per_query": _mean([float(i.num_search) for i in sub]),
        }

    overall = _block(items)

    by_source: Dict[str, Any] = {}
    sources = sorted({i.source for i in items})
    # 固定顺序排前，其余追加
    ordered = [s for s in SOURCE_ORDER if s in sources] + [
        s for s in sources if s not in SOURCE_ORDER
    ]
    for s in ordered:
        by_source[s] = _block([i for i in items if i.source == s])

    return {"overall": overall, "by_source": by_source}


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def build_markdown_table(agg: Dict[str, Any]) -> str:
    """把汇总结果渲染成一张 Markdown 指标表。"""
    by_source: Dict[str, Any] = agg["by_source"]
    src_cols = [s for s in SOURCE_ORDER if s in by_source] + [
        s for s in by_source if s not in SOURCE_ORDER
    ]

    header = "| 指标 | Overall | " + " | ".join(src_cols) + " |"
    sep = "|---|---|" + "---|" * len(src_cols)
    lines = [header, sep]

    for key, label, kind in METRIC_ORDER:
        overall_v = agg["overall"].get(key, 0.0)
        if kind == "pct":
            row = f"| {label} | {_fmt_pct(overall_v)} | "
        else:
            row = f"| {label} | {overall_v:.2f} | "
        cells = []
        for s in src_cols:
            v = by_source[s].get(key, 0.0)
            cells.append(_fmt_pct(v) if kind == "pct" else f"{v:.2f}")
        row += " | ".join(cells) + " |"
        lines.append(row)

    # 样本数一行，便于核对
    n_row = "| n | " + str(agg["overall"].get("n", 0)) + " | "
    n_row += " | ".join(str(by_source[s].get("n", 0)) for s in src_cols) + " |"
    lines.append(n_row)
    return "\n".join(lines)


# ============================================================================
# compare：baseline vs trained 对比（含 +pp 差值）
# ============================================================================
def compare_metrics(baseline_path: str, trained_path: str) -> str:
    """读取两份 metrics JSON，打印对比表（百分比指标差值以 +pp 表示）。"""

    def _load(p: str) -> Dict[str, Any]:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    base = _load(baseline_path)["overall"]
    trained = _load(trained_path)["overall"]

    lines = [f"对比：baseline=`{os.path.basename(baseline_path)}` → trained=`{os.path.basename(trained_path)}`", ""]
    header = "| 指标 | Baseline | Trained | Δ |"
    sep = "|---|---|---|---|"
    lines += [header, sep]
    for key, label, kind in METRIC_ORDER:
        b = float(base.get(key, 0.0))
        t = float(trained.get(key, 0.0))
        if kind == "pct":
            delta_pp = (t - b) * 100.0
            sign = "+" if delta_pp >= 0 else ""
            lines.append(
                f"| {label} | {_fmt_pct(b)} | {_fmt_pct(t)} | {sign}{delta_pp:.1f} pp |"
            )
        else:
            delta = t - b
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {label} | {b:.2f} | {t:.2f} | {sign}{delta:.2f} |")
    return "\n".join(lines)


# ============================================================================
# Judge / 模型端点可用性探测与构建
# ============================================================================
async def _probe_judge(judge: Optional[JudgeClient]) -> bool:
    """探测 Judge 是否真的可用：发一个廉价请求，不可达则退回规则代理。"""
    if judge is None:
        return False
    try:
        v = await judge.judge_answer("probe", "probe", ["probe"])
    except Exception:
        return False
    # JudgeClient 不可达时会把原因写进 reason
    if "unreachable" in v.reason or "invalid JSON" in v.reason:
        return False
    return True


def _print_model_endpoint_hint(endpoint: str) -> None:
    """被评模型端点不可达时的中文提示（如何用 SGLang 起模型）。"""
    print(
        "\n[提示] 无法连接被评模型端点：" + endpoint + "\n"
        "请先在 GPU 机器上用 SGLang 起一个 OpenAI 兼容服务，例如：\n"
        "  python -m sglang.launch_server \\\n"
        "      --model-path /path/to/Qwen3-8B \\\n"
        "      --host 0.0.0.0 --port 30000 \\\n"
        "      --served-model-name default --tp 2\n"
        "起好后确认 `curl " + endpoint.replace("/v1", "") + "/health` 返回 ok，"
        "再重跑本评测。\n"
    )


# ============================================================================
# 主流程
# ============================================================================
async def _run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()

    # a) 加载评测集
    if not os.path.exists(args.eval_file):
        raise FileNotFoundError(
            f"找不到评测集：{args.eval_file}\n请先运行 data/build_eval_500.py 生成。"
        )
    rows = load_eval_rows(args.eval_file, limit=args.limit)
    print(f"[load] 加载评测题 {len(rows)} 条：{args.eval_file}")

    # b) 构建工具 + agent（重依赖在函数内导入，离线单测不触发）
    #    延迟导入 tool_factory（其内部 import aiohttp / requests 等）
    from ..tools import tool_factory as tf

    tool_cfg: Dict[str, Any] = {"backend": args.search_backend, "cache_enabled": True}
    search_tool = tf.build_search_tool(tool_cfg)
    open_tool = tf.build_open_tool(tool_cfg)

    from ..agent.standalone_agent import OpenAICompatAgent

    agent = OpenAICompatAgent(
        search_tool,
        open_tool,
        model=args.model_name,
        base_url=args.model_endpoint,
        max_turns=args.max_turns,
    )

    # Judge 客户端
    judge: Optional[JudgeClient] = None
    judge_available = False
    if _HAS_JUDGE_CLIENT:
        judge = JudgeClient(base_url=args.judge_base_url, model=args.judge_model)
        judge_available = await _probe_judge(judge)
    print(f"[judge] available={judge_available}（{args.judge_base_url}）")

    thresholds = EvalThresholds()

    # c) 信号量并发跑 agent.run
    sem = asyncio.Semaphore(args.concurrency)
    model_hint_shown = {"done": False}

    async def _one(row: Dict[str, Any]) -> PerItemScore:
        async with sem:
            question = row.get("question", "")
            try:
                result = await agent.run(question)
            except Exception as exc:  # 模型端点不可达等
                if not model_hint_shown["done"]:
                    _print_model_endpoint_hint(args.model_endpoint)
                    model_hint_shown["done"] = True
                # 构造一个空结果占位，保证汇总不崩
                result = StandaloneResult(
                    question=question, answer=None, full_text=""
                )
                return PerItemScore(
                    id=str(row.get("id", "")),
                    source=str(row.get("source", "unknown")),
                    question=question,
                    answer=None,
                    gold_answers=list(row.get("gold_answers") or []),
                    num_search=0, num_open=0, num_duplicate=0, num_tool_calls=0,
                    duplicate_rate=0.0,
                    answer_correct=False, evidence_sufficient=False,
                    correct_and_sufficient=False,
                    answer_method="error", evidence_method="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            return await score_one(row, result, judge, judge_available, thresholds)

    print(f"[run] 并发={args.concurrency}，max_turns={args.max_turns}，开始推理…")
    items: List[PerItemScore] = await asyncio.gather(*[_one(r) for r in rows])

    # e) 汇总
    agg = aggregate_metrics(items)
    elapsed = time.time() - t0
    agg["overall"]["elapsed_sec"] = round(elapsed, 1)

    # f) 写 JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "config": {
            "eval_file": args.eval_file,
            "model_endpoint": args.model_endpoint,
            "model_name": args.model_name,
            "judge_base_url": args.judge_base_url,
            "judge_model": args.judge_model,
            "judge_available": judge_available,
            "max_turns": args.max_turns,
            "concurrency": args.concurrency,
            "limit": args.limit,
            "f1_correct_threshold": thresholds.f1_correct,
            "evidence_sufficient_threshold": thresholds.evidence_sufficient,
        },
        **agg,
        "per_item": [i.to_dict() for i in items],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 轨迹落盘（可选）：把完整 messages / tool_calls 写 jsonl
    if args.save_trajectories:
        traj_path = os.path.join(
            os.path.dirname(os.path.abspath(args.out)),
            "trajectories_" + os.path.basename(args.out).replace(".json", ".jsonl"),
        )
        with open(traj_path, "w", encoding="utf-8") as f:
            for it, row in zip(items, rows):
                rec = {
                    "id": it.id,
                    "source": it.source,
                    "question": it.question,
                    "gold_answers": it.gold_answers,
                    "answer": it.answer,
                    "answer_correct": it.answer_correct,
                    "evidence_sufficient": it.evidence_sufficient,
                    "num_search": it.num_search,
                    "num_open": it.num_open,
                    "num_duplicate": it.num_duplicate,
                    "error": it.error,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[save] 轨迹已写：{traj_path}")

    # 关闭 Judge 连接池
    if judge is not None:
        try:
            await judge.close()
        except Exception:
            pass

    # 控制台 Markdown 表
    md = build_markdown_table(agg)
    print("\n=== 500 题冻结评测指标 ===\n")
    print(md)
    print(f"\n[done] 耗时 {elapsed:.1f}s，指标已写：{args.out}")
    return payload


# ============================================================================
# CLI
# ============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeepSearch-RL 500 题冻结评测")
    p.add_argument("--eval_file", default="data/eval_hard_500.jsonl",
                   help="冻结评测集 jsonl（默认 data/eval_hard_500.jsonl）")
    p.add_argument("--model_endpoint", default="http://127.0.0.1:30000/v1",
                   help="被评模型的 SGLang/vLLM OpenAI 端点")
    p.add_argument("--model_name", default="default",
                   help="被评模型 served-model-name（默认 default）")
    p.add_argument("--judge_base_url", default="http://127.0.0.1:8001/v1",
                   help="Judge 服务 OpenAI 端点")
    p.add_argument("--judge_model", default="judge", help="Judge 模型名")
    p.add_argument("--search_backend", default="ddg",
                   help="搜索后端 serper/serpapi/bing/brave/tavily/ddg（默认 ddg）")
    p.add_argument("--max_turns", type=int, default=6, help="每题最多检索轮次")
    p.add_argument("--concurrency", type=int, default=16, help="并发题数")
    p.add_argument("--limit", type=int, default=None,
                   help="调试用：只跑前 N 题（默认全量 500）")
    p.add_argument("--out", default=None, help="指标 JSON 输出路径")
    p.add_argument("--save_trajectories", action="store_true",
                   help="是否把每题完整轨迹另存为 jsonl")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "TRAINED"), default=None,
                   help="对比两份 metrics JSON：--compare baseline.json trained.json")
    return p


async def evaluate_main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """异步主入口；返回写入的 payload。"""
    args = build_arg_parser().parse_args(argv)

    # compare 子能力：不跑评测，只对比两份结果
    if args.compare:
        print(compare_metrics(args.compare[0], args.compare[1]))
        return {}

    # 默认输出路径：outputs/eval/metrics_<时间戳>.json
    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join("outputs", "eval", f"metrics_{ts}.json")

    return await _run_eval(args)


def main() -> None:
    """同步 CLI 入口（被 python -m 调用）。"""
    asyncio.run(evaluate_main())


if __name__ == "__main__":
    main()
