<div align="center">

# DeepSearch-RL

### 基于 Tool-Agentic RL 的多跳搜索智能体

**Qwen3-8B · veRL · SGLang · GRPO · Agentic RL**

一个**不 fork veRL**、开箱即用的多轮搜索强化学习工程：模型在 rollout 中通过 `<search>` / `<open>` 自主检索网页，
以「证据充分度驱动的分层奖励」联合优化答案正确性、证据充分性、格式完整性与工具效率，
在 500 题冻结 Hard Multi-hop Search 评测集上把 Qwen3-8B 的准确率从 **27.8% 提升到 49.2%**。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%20cu128-orange.svg)]()
[![veRL](https://img.shields.io/badge/veRL-v0.6.0-red.svg)]()

</div>

---

## 目录

- [一、项目简介](#一项目简介)
- [二、核心特性](#二核心特性)
- [三、效果指标（复现口径）](#三效果指标复现口径)
- [四、系统架构](#四系统架构)
- [五、目录结构](#五目录结构)
- [六、环境搭建（AutoDL 4×5090）](#六环境搭建autodl-45090)
- [七、数据准备](#七数据准备)
- [八、启动检索服务](#八启动检索服务)
- [九、启动远程 Judge](#九启动远程-judge)
- [十、SwanLab 登录](#十swanlab-登录)
- [十一、启动训练](#十一启动训练)
- [十二、评估](#十二评估)
- [十三、显存与调参](#十三显存与调参)
- [十四、常见问题 FAQ](#十四常见问题-faq)
- [十五、参考与致谢](#十五参考与致谢)

---

## 一、项目简介

DeepSearch-RL 让一个基座大模型（Qwen3-8B）通过**强化学习**学会「带着工具做多跳搜索」：
面对一个需要多步推理的复杂问题，模型需要自己决定**何时搜索、搜什么、打开哪个网页、是否已经拿到足够证据、何时作答**。

与直接用 RAG 或让模型一次性输出答案不同，本工程的关键是把 **Search/Open 工具交互**纳入 RL 训练闭环：

- rollout 阶段模型与工具进行**多轮异步交互**（veRL `ToolAgentLoop` 状态机 + 自定义 `search_xml` 解析器）；
- 奖励由**远程 vLLM Judge** 构造，包含答案正确性与证据充分性；
- 奖励是「**证据充分度驱动的分层奖励**」：猜对但没有证据只能拿到低分（抑制 reward hacking），
  证据充分前的有效探索不被惩罚（缓解 under-search），重复调用与证据充分后的冗余调用被惩罚（抑制 over-search）。

整个工程只写代码、不绑定某个固定搜索服务：在线搜索支持 Serper / SerpAPI / Bing / Brave / Tavily，
无 key 时可回退免费 DuckDuckGo；工具链路带持久缓存、API-key 轮换、失败重试与异常分类，保障长时间在线 rollout 稳定。

---

## 二、核心特性

- **veRL 原生 Agentic RL**：基于 veRL v0.6.0 搭建多轮 Search/Open 工具交互链路，异步 `ToolAgentLoop` + GRPO 策略优化，无需 fork 框架。
- **证据充分度驱动的分层奖励**：联合优化答案正确性、证据充分性、格式完整性、工具效率；
  保护有效探索、惩罚重复调用及证据充分后的冗余调用，缓解 under-search 与 reward hacking。
- **远程 vLLM Judge**：独立部署 Answer Judge / Evidence Judge，训练奖励通过 HTTP 异步调用，与训练解耦。
- **工程化工具链路**：SQLite 持久缓存（多进程 WAL）、API-key 轮换池（限流冷却）、指数退避重试、异常显式分类。
- **动态 sequence balancing**：veRL 动态 batch（`use_dynamic_bsz`），将多卡 token 负载不均衡从 **52.8% 降至 0.06%**。
- **SwanLab 全链路可观测**：loss / KL / 奖励分量 / 工具调用 / 错误率 / 轨迹长度，以及完整生成样例。
- **冻结评测集与一键脚本**：500 题 Hard Multi-hop Search，输出准确率、Evidence Sufficiency、正确且证据充分占比、重复调用率、平均 Search 次数。

---

## 三、效果指标（复现口径）

评测集：**500 题冻结 Hard Multi-hop Search**（HotpotQA hard/comparison 150 + 2WikiMultihopQA 125 + MuSiQue 100 + Bamboogle 125，固定种子 seed=42）。

| 指标 | 基线 Qwen3-8B | 训练后 | 变化 |
|---|---|---|---|
| **Accuracy（准确率）** | 27.8% | **49.2%** | **+21.4 pp** |
| **Evidence Sufficiency（证据充分度）** | 22.4% | **55.8%** | **+33.4 pp** |
| **Correct & Sufficient（正确且证据充分轨迹）** | 11.0% | **41.6%** | +30.6 pp |
| **Duplicate Call Rate（重复工具调用率）** | 5.19% | **1.07%** | −4.12 pp |
| **Avg Search / query（平均搜索次数）** | — | **≈ 2.6** | — |

> 说明：上表为项目目标/复现口径。训练为在线 RL，结果会随搜索后端、数据配比、训练步数波动；
> 评估脚本同时输出各数据源（hotpotqa / 2wiki / musique / bamboogle）的分组指标，便于定位。

---

## 四、系统架构

```
                         ┌──────────────────────────────────────────────┐
                         │  训练进程（Ray + veRL，4×5090）                │
   parquet 训练集  ────▶ │  GRPO RayPPOTrainer                          │
   (56 prompts/step)     │    │                                         │
                         │    ▼                                         │
                         │  DeepSearchAgentLoop（deepsearch_agent）       │
                         │    ├─ SGLang 生成（TITO token 层拼接）          │
                         │    ├─ search_xml_parser 解析 <search>/<open>  │
                         │    └─ veRL 工具 wrapper（search/open）          │
                         │         │ HTTP                                  │
                         └─────────┼────────────────────────────────────┘
                                   ▼
                  ┌──────────────────────────────────┐
                  │ 检索服务（独立 FastAPI 进程，:8000）│
                  │  /retrieve   /open   /health      │
                  │  Serper/SerpAPI/Bing/Brave/Tavily │
                  │  持久缓存 + key 轮换 + 重试 + 分类  │
                  └──────────────────────────────────┘
                                   ▲
  远程 vLLM Judge（:8001）──────────┘  judge_client（异步 OpenAI 协议）
  · Answer Judge   · Evidence Judge
```

**一次训练 step 的流程：**

1. 从训练 parquet 采样 56 个 prompt，每个 prompt 采样 7 条 → 并行生成 **392 条多轮交互轨迹**；
2. 每条轨迹在 SGLang 上生成，`search_xml_parser` 解析 `<search>/<open>` 标签；
3. 工具 wrapper 通过 HTTP 调用检索服务（命中缓存则直接返回），观测以 `<observation>` 拼回，进入下一轮；
4. 模型给出 `<answer>` 或达到轮数上限后结束；
5. reward 函数对整条轨迹：EM/F1 判答案（必要时 Answer Judge）、Evidence Judge 判证据充分度，
   按分层公式合成奖励；
6. GRPO 用组内相对优势做策略更新；工具返回 token 通过 `response_mask=0` 不参与策略梯度。

> **TITO（Token-In-Token-Out）**：多轮拼接在 token 层完成，绝不 decode 成文字再重新 encode，
> 否则会导致轨迹脱离策略分布、PPO 不收敛。这部分由 veRL 框架保证，我们只在配置层启用。

---

## 五、目录结构

```
DeepSearch-RL/
├── README.md                     # 本文档
├── requirements.txt              # 通用依赖（torch/verl/sglang 需特殊安装）
├── setup.py
├── run.sh                        # 一键编排（retrieval/judge/train/eval/install/model）
├── configs/
│   ├── grpo_qwen3_8b_4x5090.yaml # 主训练配置（默认 4×5090）
│   ├── tools_search_xml.yaml     # veRL 工具注册（search/open wrapper）
│   ├── judge.yaml                # Judge 配置
│   └── eval_500.yaml             # 评估配置
├── scripts/
│   ├── install_autodl.sh         # AutoDL 一键装环境
│   ├── download_model.sh         # 下载 Qwen3-8B（ModelScope）
│   ├── download_data.sh          # 下载并预处理数据
│   ├── start_retrieval.sh        # 启动检索服务
│   ├── start_judge.sh            # 启动 Judge
│   ├── train.sh                  # 启动训练
│   └── eval.sh                   # 启动评估
├── data/
│   ├── download_data.py          # 下载 5 个数据源并归一
│   ├── prepare_train.py          # 生成 veRL 训练 parquet
│   ├── build_eval_500.py         # 构造 500 题冻结评测集
│   └── README.md
├── deepsearch_rl/
│   ├── protocol.py               # 工具标签协议（解析/观测/系统提示）
│   ├── tools/                    # 搜索/网页工具、缓存、key 轮换、异常、veRL wrapper
│   ├── retrieval/                # FastAPI 检索服务
│   ├── judge/                    # Judge 提示词、启动器、客户端
│   ├── agent/                    # 解析器、自定义 loop、轨迹分析、独立推理
│   ├── rewards/                  # EM/F1、工具效率、分层奖励
│   ├── train/                    # 训练入口、SwanLab 封装
│   ├── eval/                     # 500 题评估
│   └── utils/                    # 配置/日志/随机种子
└── tests/                        # 离线逻辑测试（不触网）
```

---

## 六、环境搭建（AutoDL 4×5090）

### 6.1 租机与镜像

- 在 AutoDL 租用 **4×RTX 5090（32GB，Blackwell sm_120）**；
- 镜像选择 **Ubuntu 22.04/24.04 + Python 3.12 + CUDA 12.8**；
- 5090 是 Blackwell 新架构，**必须使用 CUDA 12.8 + PyTorch cu128 车道**（cu126 及以下不含 sm_120 kernel）。

### 6.2 获取工程

```bash
git clone https://github.com/Simon11866/DeepSearch-RL.git
cd DeepSearch-RL
pip install -e .
```

### 6.3 一键安装（推荐）

```bash
bash scripts/install_autodl.sh
```

该脚本会依次完成：PyTorch 2.8.0 cu128 → 通用依赖 → veRL v0.6.0（源码 editable，`[sglang]`）
→ 固定 sglang ≤0.5.19 + flashinfer cu128 → liger-kernel。可重复执行。

### 6.4 手动安装（如需逐步控制）

```bash
# 1) PyTorch 2.8.0 cu128（5090 必须）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 2) 通用依赖
pip install -r requirements.txt

# 3) veRL v0.6.0（必须 pin，main 已迁 CUDA13/torch2.14）
git clone https://github.com/volcengine/verl.git ~/verl
cd ~/verl && git checkout v0.6.0
pip install -e ".[sglang]"
cd -

# 4) sglang 固定 cu12 车道（0.5.19 是最后一个 CUDA12 版本，勿升 0.5.20+）
pip install "sglang>=0.4.6.post1,<0.5.20"
pip install flashinfer_python \
  --find-links https://flashinfer.ai/whl/cu128/torch2.8/flashinfer-python

# 5) 免编译 kernel（5090 上替代 flash-attn）
pip install "liger-kernel>=0.8.2"
```

> **关于 flash-attn**：FlashAttention-3/4 面向数据中心 Blackwell（sm_100，带 TMEM），**在桌面 5090（sm_120）上无法运行**；
> 如确需 FlashAttention-2，需 `flash-attn>=2.7.2`（含 sm_120 kernel）且本机 nvcc=12.8。
> 本工程默认用 **liger-kernel + SDPA**（训练）与 SGLang 的 `triton/flashinfer` 后端（推理），免编译。

### 6.5 关键依赖版本一览

| 包 | 版本 | 备注 |
|---|---|---|
| Python | 3.12 | |
| CUDA | 12.8 | 5090 必须 |
| torch / torchvision / torchaudio | 2.8.0 / 0.23.0 / 2.8.0 | cu128 index |
| verl | **v0.6.0** | 源码 editable，勿用 main |
| sglang | **≥0.4.6, <0.5.20** | 最后 CUDA12 车道 |
| flashinfer | cu128/torch2.8 | 对应 find-links |
| vllm（Judge 服务） | 随 verl v0.6.0 | OpenAI 兼容端点 |
| liger-kernel | ≥0.8.2 | 免编译 kernel |
| modelscope | 1.23.1 | 模型/数据下载 |
| ray | ≥2.45, <2.49 | 分布式 |
| swanlab | 0.9.0 | 实验追踪 |

---

## 七、数据准备

### 7.1 下载模型（Qwen3-8B，ModelScope）

```bash
bash scripts/download_model.sh
# 下载完成后：
export MODEL_PATH=$HOME/models/Qwen3-8B
```

等价的手动下载：

```bash
modelscope download --model Qwen/Qwen3-8B --local_dir $HOME/models/Qwen3-8B
# 或
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B'))"
```

### 7.2 下载并预处理数据

```bash
# 国内（ModelScope 优先，HF 自动走 hf-mirror 镜像）
bash scripts/download_data.sh
```

该脚本依次执行：

1. **下载**：HotpotQA（fullwiki）、2WikiMultihopQA、MuSiQue、NQ-open、Bamboogle，归一为统一中间 jsonl；
2. **构造 500 题冻结评测集**：`data/eval_hard_500.jsonl`（seed=42，配额 150/125/100/125）；
3. **生成训练 parquet**：`data/processed/train/`、`data/processed/val/`。

单独执行各步：

```bash
python data/download_data.py --out_dir data/raw --nq_limit 30000
python data/build_eval_500.py --raw_dir data/raw --out data/eval_hard_500.jsonl
python data/prepare_train.py --raw_dir data/raw --out_dir data/processed
```

**训练集口径**：NQ-open 降采样 30k + HotpotQA fullwiki train + 2Wiki train + MuSiQue train；Bamboogle 仅用于评测、不入训练。

训练 parquet 每行字段：

```json
{
  "data_source": "nq|hotpotqa|2wiki|musique",
  "prompt": [{"role": "system", "content": "<系统提示>"}, {"role": "user", "content": "<问题>"}],
  "ability": "multi-hop-search",
  "agent_name": "deepsearch_agent",
  "reward_model": {"style": "rule", "ground_truth": {"target": ["答案", "别名..."]}},
  "extra_info": {"split": "train", "index": 0, "num_hops": 2}
}
```

> 数据集来源：HotpotQA `hotpotqa/hotpot_qa`、2WikiMultihopQA `voidful/2WikiMultihopQA`、
> MuSiQue `dgslibisey/MuSiQue`、NQ-open `google-research-datasets/nq_open`、Bamboogle（ofirpress/self-ask）。

---

## 八、启动检索服务

**终端 A（常驻）**：

```bash
# 有付费 key（推荐，稳定）：以 Serper 为例
export SERPER_API_KEYS="key1,key2"      # 多个 key 逗号分隔，自动轮换
bash scripts/start_retrieval.sh

# 或使用其它后端
SEARCH_BACKEND=serpapi SERPAPI_API_KEYS="key" bash scripts/start_retrieval.sh
SEARCH_BACKEND=bing    BING_API_KEYS="key"    bash scripts/start_retrieval.sh
SEARCH_BACKEND=tavily  TAVILY_API_KEYS="key"  bash scripts/start_retrieval.sh

# 无 key：自动/显式回退免费 DuckDuckGo（仅建议调试）
SEARCH_BACKEND=ddg bash scripts/start_retrieval.sh
```

服务默认监听 `http://0.0.0.0:8000`，提供：

| 接口 | 入参 | 出参 |
|---|---|---|
| `POST /retrieve` | `{"queries":[...], "topk":5, "return_scores":true}` | `{"result": [[{"document":{"title","url","contents"},"score"}]]}` |
| `POST /open` | `{"url":"..."}` | `{"url","contents","ok"}` |
| `GET /health` | — | `{"status":"ok","cache":{...},"keys_available":n}` |

可用环境变量：`RETRIEVAL_PORT`（默认 8000）、`RETRIEVAL_CONCURRENCY`（默认 120）、
`CACHE_DB_PATH`（默认 `~/.cache/deepsearch_rl/tool_cache.db`）、`KEY_FILE`、`UVICORN_WORKERS`。

---

## 九、启动远程 Judge

**终端 B（常驻）**：

```bash
# 默认用 Qwen3-8B 作为裁判模型，端口 8001
bash scripts/start_judge.sh

# 自定义裁判模型 / 并行度
JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct JUDGE_TP=1 JUDGE_GPU_MEM=0.4 \
  bash scripts/start_judge.sh
```

等价的手动 vLLM 命令：

```bash
vllm serve Qwen/Qwen3-8B \
  --served-model-name judge \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.4 \
  --max-model-len 8192 --dtype bfloat16 --enable-prefix-caching
```

> **注意模型名**：启动器用了 `--served-model-name judge`，因此 reward/eval 客户端的 `JUDGE_MODEL` 要设为 **`judge`**
> （主配置默认已是 `judge`）；若不使用 `--served-model-name`，则客户端用模型路径名。

两类裁判：

- **Answer Judge**：对照 gold answers 判断预测答案是否正确（correct / partial / wrong）；
- **Evidence Judge**：判断收集到的证据是否足以支撑答案，输出 0..1 连续分（阈值 0.6 判充分）。

---

## 十、SwanLab 登录

```bash
pip install swanlab==0.9.0

# 方式一：交互式登录（API key 在 https://swanlab.cn/settings 获取）
swanlab login

# 方式二：非交互（CI / 无 TTY）
export SWANLAB_API_KEY=xxxxxxxx
```

> veRL v0.6.0 **已内置 SwanLab 追踪**，配置里 `trainer.logger=[console,swanlab]` 即可，无需额外适配。
> 无网/不上传时：`export SWANLAB_MODE=offline`（日志落本地，可后续 `swanlab watch` 同步）。

SwanLab 上可看到：`actor/policy_loss`、`actor/kl`、`actor/entropy`、`actor/grad_norm`、
`reward/score` 及 format/answer/evidence/tool 各分量、`tool/num_search`、`tool/duplicate_rate`、
`tool/error_rate`、`rollout/num_turns`、`timing/step_sec`，以及周期性记录的完整轨迹文本。

---

## 十一、启动训练

**终端 C**（确保终端 A、B 已启动）：

```bash
export MODEL_PATH=$HOME/models/Qwen3-8B
export SWANLAB_API_KEY=xxxx          # 或 export SWANLAB_MODE=offline
bash scripts/train.sh
```

训练入口支持 `--dry_run`（只打印最终 Hydra 覆盖、不真正训练），便于先核对配置：

```bash
MODEL_PATH=$HOME/models/Qwen3-8B \
  python -m deepsearch_rl.train.train_grpo --dry_run
```

### 默认配置口径（4×5090）

- `data.train_batch_size=56`（56 prompts），`rollout.n=7` → **56×7 = 392 条多轮轨迹**；
- `rollout.name=sglang`、`rollout.mode=async`、`multi_turn.enable=True`、`multi_turn.format=search_xml`；
- `agent.default_agent_loop=deepsearch_agent`；
- `actor.use_dynamic_bsz=True`（动态 sequence balancing）、`state_masking=True`；
- `actor.use_kl_loss=True`、`kl_loss_type=low_var_kl`、`kl_loss_coef=0.001`；
- `algorithm.adv_estimator=grpo`、`temperature=1.0`、`lr=1e-6`、constant 调度。

### 命令行覆盖（Hydra key=value）

```bash
bash scripts/train.sh \
  data.train_batch_size=128 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45
```

---

## 十二、评估

训练完成后，先用 SGLang/vLLM 把**被评模型**以 OpenAI 端点起好（例如端口 30000），再跑评测：

```bash
# 终端里先起被评模型（示例：SGLang）
python -m sglang.launch_server \
  --model $HOME/checkpoints/deepsearch-rl/qwen3-8b/actor \
  --served-model-name default --port 30000

# 全量 500 题（Judge 服务需在 :8001 运行）
bash scripts/eval.sh

# 调试：只跑前 20 题
bash scripts/eval.sh --limit 20

# 保存每题完整轨迹
bash scripts/eval.sh --save_trajectories
```

输出 5 个核心指标 + 按数据源分组，并写 JSON 到 `outputs/eval/`。

**基线 vs 训练后对比**：

```bash
bash scripts/eval.sh --compare outputs/eval/metrics_baseline.json outputs/eval/metrics_trained.json
```

---

## 十三、显存与调参

### 13.1 显存估算（4×5090 32GB）

- **训练态**：FSDP 分片 bf16 参数 + bf16 梯度 + fp32 Adam 状态约 **24GB/卡**，配合梯度检查点与动态 batch，可在 32GB 内；
- **Rollout 态**：SGLang 默认 `TP=2 / DP=2`，权重分片 + KV 缓存由 `gpu_memory_utilization`（默认 0.55）控制；
- 训练与 rollout 不同时占用全部资源（hybrid engine 在阶段切换时 offload/释放）。

### 13.2 显存紧张时（按优先级）

1. 降低 `rollout.n`（7→5），直接减少并发轨迹数；
2. 降低 `gpu_memory_utilization`（0.55→0.45）；
3. 降低 `actor.ppo_micro_batch_size_per_gpu`（4→2）、`ppo_max_token_len_per_gpu`；
4. 缩短 `max_response_length` / `max_tool_response_length`；
5. 开启 actor CPU offload（`actor_rollout_ref.actor` 对应 offload 项），用显存换速度。

### 13.3 关键调参建议

| 现象 | 建议 |
|---|---|
| 模型不搜索、直接猜（under-search） | 降低 answer gate 下限/提高证据门控权重；适当提高 `temperature`；确认证据奖励生效 |
| 模型反复刷搜索（over-search） | 提高重复/冗余惩罚；降低 `efficient_budget`；检查证据充分奖励是否过早饱和 |
| 奖励虚高、答案却错（reward hacking） | 强化答案门控（无证据正确只给低分）；Answer Judge 与 EM 双重校验 |
| 收敛慢 / 波动大 | 组大小 `n` 取 5~8；`kl_loss_coef` 在 0.001~0.01 调整；lr 1e-6 量级 |
| 多卡负载不均 | 保持 `use_dynamic_bsz=True`；检查超长样本是否被过滤 |

---

## 十四、常见问题 FAQ

**Q1：为什么必须用 cu128 / torch 2.8.0？**
RTX 5090 是 Blackwell sm_120，cu126 及以下 wheel 不含该架构 kernel，会报 `no kernel image available`。
务必用 `--index-url https://download.pytorch.org/whl/cu128`。

**Q2：为什么 pin verl v0.6.0、sglang ≤0.5.19？**
verl main 与 sglang 0.5.20+ 已迁移到 CUDA 13 / torch 2.13+，与本工程 cu128/torch2.8 车道冲突，
会拉到无法在 5090 上运行的依赖。

**Q3：不买搜索 API 能跑吗？**
可以，`SEARCH_BACKEND=ddg` 使用免费 DuckDuckGo（无需 key），但稳定性与召回不如付费服务，建议仅用于调试。

**Q4：Judge 服务一定要单独起吗？**
奖励函数通过 HTTP 调用 Judge，需要它常驻。Judge 不可达时 reward 会自动降级为纯规则（EM/F1 + 标题命中），
训练不会中断，但证据充分度信号会变弱。

**Q5：工具返回的内容会参与策略梯度吗？**
不会。veRL 对工具返回 token 自动标 `response_mask=0`，策略梯度只在模型自己生成的 token 上计算。

**Q6：如何调整并行度适配不同卡数？**
- 8 卡：`trainer.n_gpus_per_node=8`，可把 `rollout.data_parallel_size` 提到 4、`train_batch_size` 提到 112；
- 2 卡：`n_gpus_per_node=2`，`rollout.tensor_model_parallel_size=2, data_parallel_size=1`，`train_batch_size=28`。

---

## 十五、参考与致谢

本工程在设计与实现上借鉴了以下优秀项目（均为实际调研）：

- [veRL](https://github.com/volcengine/verl)：火山引擎开源 RLHF/Agentic RL 框架，`ToolAgentLoop`、GRPO、动态 batch；
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)：搜索/答案标签协议、EM 奖励、检索服务；
- [R1-Searcher](https://github.com/PeterGriffinJin/R1-Searcher)：检索奖励与训练数据组织；
- [SimpleRL-Zoo](https://github.com/SimpleAI-Zoo/SimpleRL-Zoo)：异步 rollout 与 weight-sync；
- 数据集：HotpotQA、2WikiMultihopQA、MuSiQue、Natural Questions、Bamboogle；基座模型：Qwen3-8B。

## License

本项目基于 [Apache License 2.0](LICENSE) 发布。
