# -*- coding: utf-8 -*-
"""
DeepSearch-RL 主训练入口（GRPO / veRL / SGLang / 4×5090）
======================================================

职责：把工程内「可读的嵌套 yaml」（configs/grpo_qwen3_8b_4x5090.yaml）翻译成
veRL 自带的 Hydra 配置（``ppo_trainer``），再调用 ``verl.trainer.main_ppo.run_ppo``。

为什么要这一层包装
------------------
veRL v0.6.0 的配置是 Hydra 组合出来的，命令行习惯是 ``key=value`` 覆盖；而我们为了
人读方便，维护了一份嵌套 yaml。本脚本做三件事：

1. 用 OmegaConf 加载嵌套 yaml，把末尾两个「非 veRL 段」（``retrieval`` / ``judge``）
   拆出来，转成环境变量（RETRIEVAL_SERVICE_URL / RETRIEVAL_CONCURRENCY /
   JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_SUFFICIENCY_THRESHOLD）——这些是工具 wrapper
   与 reward 函数在 worker 进程里直接读的，不走 Hydra。
2. 把剩下的 veRL 原生嵌套配置（trainer/data/actor_rollout_ref/algorithm/
   reward_model/custom_reward_function）拍平成 Hydra dotlist 覆盖项。
3. 先 ``import deepsearch_rl.agent.deepsearch_loop``（触发 ``deepsearch_agent`` 与
   ``search_xml`` 两个注册），再用 ``hydra.compose`` 组合 ppo_trainer，最后
   ``verl.trainer.main_ppo.run_ppo(config)``。

用法
----
.. code-block:: bash

    # 用默认配置训练（先 export MODEL_PATH；retrieval/judge 服务已在 8000/8001 起好）
    python -m deepsearch_rl.train.train_grpo

    # 指定配置 + 任意 Hydra 覆盖（key=value 直接跟在后面即可）
    python -m deepsearch_rl.train.train_grpo \
        --config configs/grpo_qwen3_8b_4x5090.yaml \
        data.train_batch_size=128 actor_rollout_ref.rollout.n=5

    # 只看最终会下发给 veRL 的覆盖项与解析后配置，不真正起训练（本地/CI 用）
    python -m deepsearch_rl.train.train_grpo --dry_run

设计约束：顶层不 import hydra / verl / torch，保证 ``py_compile`` 与 ``--dry_run``
在无 verl 环境下也能跑；重依赖一律在 ``main()`` 真正训练时惰性导入。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, List, Tuple

# yaml 与 Hydra 覆盖项构造只用 omegaconf（轻量、纯 python），本地无 verl 也能 dry-run
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# 非 veRL 段 → 环境变量 的映射表（与 yaml 末尾 retrieval/judge 段一一对应）
# ---------------------------------------------------------------------------
_RETRIEVAL_ENV_MAP = {
    # yaml 里的键 -> 环境变量名
    "service_url": "RETRIEVAL_SERVICE_URL",
    "concurrency": "RETRIEVAL_CONCURRENCY",
}
_JUDGE_ENV_MAP = {
    "base_url": "JUDGE_BASE_URL",
    "model": "JUDGE_MODEL",
    "sufficiency_threshold": "JUDGE_SUFFICIENCY_THRESHOLD",
}

# 路径类列表键：逐项拍平为 Hydra 列表元素覆盖（+key.0=...、+key.1=...），
# 而不是内联成 [a,b]。这样 CLI 上再追加一条数据目录也很直观。
_PATH_LIST_KEYS = ("train_files", "val_files", "test_files")


# ---------------------------------------------------------------------------
# 基础工具：标量 / Hydra 值字符串化
# ---------------------------------------------------------------------------
def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _format_scalar(v: Any) -> str:
    """把 python 标量格式成 Hydra override 右侧的字面量。

    - None -> ``null``
    - bool -> ``true`` / ``false``（Hydra 小写）
    - int/float -> repr（1e-6 这类科学计数法 Hydra 也能解析）
    - str -> 不含 Hydra 特殊字符时裸写；否则加双引号转义
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    # 含空白或 Hydra 语法字符（空格/逗号/方括号/等号/冒号/$）时加引号
    if re.search(r'[\s\[\]\{\}=,$:]', s):
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return s


def _flatten(cfg: dict, prefix: str = "") -> List[Tuple[str, str]]:
    """把嵌套 dict 递归拍平成 ``(dot_key, override_value)`` 列表。

    规则：
    - 嵌套 dict -> 继续下钻（``a.b.c``）；
    - 标量 -> 直接 ``key=value``；
    - 标量列表：
        * 路径类键（train_files/val_files/test_files）-> 逐项 ``+key.0=v0``、``+key.1=v1``
          （Hydra 逐元素覆盖语法，可在空列表/None 默认值上创建列表）；
        * 其余（如 ``trainer.logger=[console,swanlab]``）-> 内联 ``key=[v0,v1]``；
    - 非标量列表（dict 列表）：本工程主 yaml 中不存在（工具挂载走外部
      ``tool_config_path``），遇到时打印警告并跳过，避免拍出错位的键。
    """
    out: List[Tuple[str, str]] = []
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten(v, key))
        elif isinstance(v, list):
            if not v:
                continue
            if all(_is_scalar(x) for x in v):
                if key.split(".")[-1] in _PATH_LIST_KEYS:
                    # 路径列表：逐项 +key.{idx}=item
                    for idx, item in enumerate(v):
                        out.append((f"+{key}.{idx}", _format_scalar(item)))
                else:
                    # 普通标量列表：内联 [a,b]
                    out.append((key, "[" + ",".join(_format_scalar(x) for x in v) + "]"))
            else:
                print(
                    f"[warn] 复杂（dict）列表未拍平，已跳过：{key}；"
                    "如确需覆盖请改用 CLI key=value 传入。",
                    file=sys.stderr,
                )
        else:
            out.append((key, _format_scalar(v)))
    return out


def _extract_env_sections(cfg: dict) -> dict:
    """拆出 retrieval / judge 两个非 veRL 段并据此设置环境变量，返回这两个段的内容。"""
    retrieval = cfg.pop("retrieval", {}) or {}
    judge = cfg.pop("judge", {}) or {}

    for yaml_key, env_name in _RETRIEVAL_ENV_MAP.items():
        if yaml_key in retrieval:
            os.environ[env_name] = str(retrieval[yaml_key])
    for yaml_key, env_name in _JUDGE_ENV_MAP.items():
        if yaml_key in judge:
            os.environ[env_name] = str(judge[yaml_key])

    return {"retrieval": retrieval, "judge": judge}


def _check_model_path(dry_run: bool) -> None:
    """检查 MODEL_PATH 是否已设置；缺失时给出清晰中文报错（dry-run 用占位值继续）。"""
    mp = os.environ.get("MODEL_PATH", "").strip()
    if mp:
        return
    msg = (
        "[错误] 未检测到环境变量 MODEL_PATH。\n"
        "       model.path 在 yaml 里写作 ${oc.env:MODEL_PATH}，必须先指定本地模型目录。\n"
        "       解决办法二选一：\n"
        "         1) 先下载模型： bash scripts/download_model.sh\n"
        "         2) 直接 export： export MODEL_PATH=/path/to/Qwen3-8B\n"
    )
    if dry_run:
        # dry-run 不真正加载模型，给一个占位值，让 dotlist 能打印出来
        print(msg + "       [dry-run] 已临时使用占位值 ./models/Qwen3-8B 继续。", file=sys.stderr)
        os.environ["MODEL_PATH"] = "./models/Qwen3-8B"
    else:
        print(msg, file=sys.stderr)
        sys.exit(2)


def build_overrides(config_path: str, dry_run: bool) -> Tuple[List[str], dict, List[str]]:
    """加载 yaml、拆 env 段、拍平成 Hydra dotlist。

    返回 (dotlist, env_sections, veRL_nested_dict)。
    dotlist 形如 ``["actor_rollout_ref.rollout.n=7", "+data.train_files.0=data/processed/train", ...]``。
    """
    if not os.path.isfile(config_path):
        print(f"[错误] 配置文件不存在：{config_path}", file=sys.stderr)
        sys.exit(2)

    # 先确保 MODEL_PATH（resolve ${oc.env:MODEL_PATH} 时需要）
    _check_model_path(dry_run)

    cfg = OmegaConf.load(config_path)
    # resolve=True 会把 ${oc.env:MODEL_PATH} 解析成本地路径
    container: dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    if not isinstance(container, dict):
        print(f"[错误] 配置根节点不是 mapping：{config_path}", file=sys.stderr)
        sys.exit(2)

    # 1) 拆 retrieval/judge -> 环境变量
    env_sections = _extract_env_sections(container)

    # 2) 拍平剩余 veRL 原生段
    flat = _flatten(container)
    dotlist = [f"{k}={v}" for k, v in flat]
    return dotlist, env_sections, container


def parse_cli(argv: List[str]) -> argparse.Namespace:
    """解析命令行：--config / --dry_run 由本脚本处理，其余 key=value 原样透传。"""
    parser = argparse.ArgumentParser(
        description="DeepSearch-RL GRPO 训练入口（veRL ppo_trainer 包装）",
        usage="python -m deepsearch_rl.train.train_grpo [--config YAML] [--dry_run] "
        "[--] [hydra key=value ...]",
    )
    parser.add_argument(
        "--config",
        default="configs/grpo_qwen3_8b_4x5090.yaml",
        help="主配置 yaml（默认 configs/grpo_qwen3_8b_4x5090.yaml）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印最终 overrides 与解析后配置，不调用 run_ppo",
    )
    # 其余参数（含 -- 之后的）全部进 extras，作为 Hydra overrides 透传
    args, extras = parser.parse_known_args(argv)
    args.overrides = extras
    return args


def main(argv: List[str] | None = None) -> None:
    args = parse_cli(sys.argv[1:] if argv is None else argv)

    # 相对路径基于工程根目录（train_grpo 通常从工程根用 -m 调用）
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        # 兼容在任意 cwd 下调用：尝试相对于 __file__ 上跳两层（工程根）
        here = os.path.dirname(os.path.abspath(__file__))
        proj_root = os.path.abspath(os.path.join(here, "..", ".."))
        candidate = os.path.join(proj_root, config_path)
        if os.path.exists(candidate):
            config_path = candidate

    dotlist, env_sections, veRL_nested = build_overrides(config_path, args.dry_run)

    # 3) 追加用户 CLI overrides（优先级最高，放最后）
    user_overrides: List[str] = list(args.overrides)
    # 去掉 argparse 可能残留的 "--"
    user_overrides = [o for o in user_overrides if o != "--"]
    final_overrides = dotlist + user_overrides

    # ---------------------------- dry-run 输出 ----------------------------
    if args.dry_run:
        print("=" * 70)
        print("[dry-run] 配置文件：", config_path)
        print("-" * 70)
        print("[dry-run] retrieval/judge 段（已写入环境变量）：")
        for name, sec in env_sections.items():
            print(f"  {name}: {sec}")
        print("-" * 70)
        print("[dry-run] 当前相关环境变量：")
        for env_name in (
            "MODEL_PATH",
            "RETRIEVAL_SERVICE_URL",
            "RETRIEVAL_CONCURRENCY",
            "JUDGE_BASE_URL",
            "JUDGE_MODEL",
            "JUDGE_SUFFICIENCY_THRESHOLD",
            "SWANLAB_MODE",
        ):
            print(f"  {env_name}={os.environ.get(env_name, '<未设置>')}")
        print("-" * 70)
        print(f"[dry-run] 拍平 + 用户追加的 Hydra overrides（共 {len(final_overrides)} 条）：")
        for o in final_overrides:
            print("   ", o)
        print("-" * 70)
        print("[dry-run] 解析后的 veRL 嵌套配置（OmegaConf.to_container 结果）：")
        print(OmegaConf.to_yaml(veRL_nested, sort_keys=True))
        print("=" * 70)
        print("[dry-run] 未调用 hydra.compose / run_ppo。")
        return

    # ---------------------------- 真正训练 ----------------------------
    # 4) 先 import 我们的 agent loop，触发 deepsearch_agent / search_xml 注册；
    #    必须在 hydra 组合 & run_ppo 之前完成，否则 veRL 按名字查不到这两个类。
    import deepsearch_rl.agent.deepsearch_loop  # noqa: F401  （注册副作用）

    # 5) 惰性导入 hydra 与 verl（训练环境才有）
    import hydra
    from omegaconf import OmegaConf as OC  # 复用，便于打印

    with hydra.initialize_config_module(
        config_module="verl.trainer.config", version_base=None
    ):
        cfg = hydra.compose(config_name="ppo_trainer", overrides=final_overrides)

    print("=" * 70)
    print("[train] hydra compose 完成，关键生效配置：")
    print(OmegaConf.to_yaml(
        OC.create({
            "data.train_batch_size": cfg.data.train_batch_size,
            "rollout.n": cfg.actor_rollout_ref.rollout.n,
            "rollout.multi_turn.enable": cfg.actor_rollout_ref.rollout.multi_turn.enable,
            "rollout.multi_turn.format": cfg.actor_rollout_ref.rollout.multi_turn.format,
            "agent.default_agent_loop": cfg.actor_rollout_ref.rollout.agent.default_agent_loop,
            "custom_reward": cfg.custom_reward_function,
            "model.path": cfg.actor_rollout_ref.model.path,
            "logger": cfg.trainer.logger,
        }),
        sort_keys=False,
    ))
    print("=" * 70)

    # 6) 进入 veRL 主流程（ray.init / worker 注册 / load_reward_manager / fit）
    from verl.trainer.main_ppo import run_ppo

    run_ppo(cfg)


if __name__ == "__main__":
    main()
