# data/ 数据目录说明

DeepSearch-RL 的数据下载、统一中间格式、训练 parquet 与 500 题冻结评测集均在本目录生成。

## 1. 数据源

| source    | HuggingFace ID / 来源                          | config / split            | 单跳/多跳 | 说明 |
|-----------|------------------------------------------------|---------------------------|-----------|------|
| nq        | `google-research-datasets/nq_open`             | train / validation        | 单跳      | 真实 Google query，answer 为 list[str] |
| hotpotqa  | `hotpotqa/hotpot_qa`                           | `fullwiki`，train / validation | 2 跳 | answer 为 str；type=bridge/comparison；level=easy/medium/hard |
| 2wiki     | `voidful/2WikiMultihopQA`                      | train / validation        | 2 跳      | answer 为 str；带 evidences/supporting_facts |
| musique   | `dgslibisey/MuSiQue`                           | train / validation        | 2-4 跳    | answer=str + answer_aliases=list；answerable=bool |
| bamboogle | [官方 json](https://raw.githubusercontent.com/ofirpress/self-ask/master/data/bamboogle_2hop.json) | 仅 validation（125 题） | 2 跳 | 对抗题，golden_answers=list；不进训练集 |

## 2. 统一中间 jsonl 字段（download_data.py 产出到 `data/raw/`）

每个源每个 split 一个文件：`{source}_{split}.jsonl`，每行字段：

```json
{
  "id": "hotpot_xxx | nq_xxx | 2wiki_xxx | musique_xxx | bamboogle_0000",
  "source": "nq | hotpotqa | 2wiki | musique | bamboogle",
  "question": "...",
  "gold_answers": ["主答案", "别名..."],
  "type": "single-hop | bridge | comparison | compositional ...",
  "level": "easy | medium | hard",
  "num_hops": 1,
  "supporting_titles": ["标题1", "标题2"],
  "split": "train | validation"
}
```

> MuSiQue 行额外带 `answerable: bool`，供 500 题评测按 answerable 过滤。

答案归一规则：HotpotQA `[answer]`；NQ-open 直接用 `answer` 列表；MuSiQue `[answer] + answer_aliases`；2Wiki `[answer]`；Bamboogle 用 `golden_answers`。

## 3. 训练 parquet（prepare_train.py 产出到 `data/processed/`）

严格按 SPEC 4.1，每行：

```json
{
  "data_source": "nq | hotpotqa | 2wiki | musique",
  "prompt": [
    {"role": "system", "content": "<SYSTEM_PROMPT>"},
    {"role": "user",   "content": "<build_user_prompt(question)>"}
  ],
  "ability": "multi-hop-search",
  "agent_name": "deepsearch_agent",
  "reward_model": {"style": "rule", "ground_truth": {"target": ["答案", "别名..."]}},
  "extra_info": {"split": "train|val", "index": 0, "num_hops": 2}
}
```

- prompt 是 `list[dict]`（veRL `return_raw_chat=True` 要求）。
- 训练混合：NQ-open 降采样到 `--nq_limit`（默认 30000，seed=42）+ HotpotQA/2Wiki/MuSiQue train 全量；Bamboogle 排除。
- 从每个 train 源固定抽 200 条做训练期 val（seed=42）。
- 输出 shard：`data/processed/train/{source}.parquet`、`data/processed/val/{source}.parquet`。

## 4. 500 题冻结评测集（build_eval_500.py 产出 `data/eval_hard_500.jsonl`）

只用各源 validation/dev，seed=42，配额：

| 来源 split                  | 配额 | 筛选条件                          |
|----------------------------|-----:|-----------------------------------|
| HotpotQA fullwiki validation| 150  | `level==hard`；不足补 `type==comparison` |
| 2WikiMultihopQA validation  | 125  | 全类型                            |
| MuSiQue validation          | 100  | `answerable==True`                |
| Bamboogle（全集）          | 125  | 全收（只有 125）                  |
| **合计**                   | **500** |                                   |

输出字段：`id, source, question, gold_answers, type, level, num_hops, supporting_titles, split`。一次性生成后冻结，勿重抽。

## 5. 命令

```bash
# 一键：下载 -> 500 评测集 -> 训练 parquet（建议在 AutoDL 上跑）
bash scripts/download_data.sh

# 或分步：
python data/download_data.py --out_dir data/raw \
    --sources nq,hotpotqa,2wiki,musique,bamboogle \
    --nq_limit 30000 --backend auto

python data/build_eval_500.py --raw_dir data/raw \
    --out data/eval_hard_500.jsonl --seed 42

python data/prepare_train.py --raw_dir data/raw \
    --out_dir data/processed --nq_limit 30000 --seed 42
```

模型下载：

```bash
bash scripts/download_model.sh   # 默认落盘 ~/models/Qwen3-8B
```

## 6. 后端 / 网络提示

- `--backend auto`：优先 modelscope，失败回退 huggingface datasets。
- 国内 HF 加速：脚本自动设置 `HF_ENDPOINT=https://hf-mirror.com`，可手动覆盖。
- 依赖：`pip install -U datasets huggingface_hub modelscope pyarrow`。
