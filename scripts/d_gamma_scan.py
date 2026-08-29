#!/usr/bin/env python
"""D-γ 扫描脚本（Stage 2 支线）——方案 §6 机遇 3 的零训练成本诊断。

权重插值 θ(γ) = θ₀ + γ·(θ_SimNPO − θ₀)，γ ∈ {0.5, 0.75, 1.0, 1.25, 1.5}，
逐 γ 测 (D_KS, MU) 轨迹，判定 SimNPO 的效用崩溃是「漂移」（沿 S^⊥ 乱走，
可恢复）还是「真删除」（不可逆），并约束 Stage 5 的搜索空间（S 还是
S ∪ {Δ_FO}，K+1 维）。

用法（远程 2×A800 上运行，1–2 GPU-h）：
    python scripts/d_gamma_scan.py \\
        --theta-0-path open-unlearning/tofu_Llama-3.2-1B-Instruct_full \\
        --simnpo-path /path/to/saves/simnpo_forget10_ep5/checkpoint \\
        --retain-logs-path saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json

测量复用仓库指标框架（configs/experiment/eval/tofu/default.yaml）：
    - D_KS 走 Stage 1 的 ``ks_statistic`` 通道（forget_quality_D 指标），
      需要冻结 retain 模型的 TR JSON（--retain-logs-path）；缺失时输出 None
      （报告为「—」），不臆造数据。
    - MU 走 ``model_utility``（hm_aggregate，retain/ra/wf 三组）。

🔴 纪律（主方案 §11 事实 2）：D 口径只用 KS 统计量，不用 forget_quality 的
p 值；不写任何「SimNPO 原文 FQ=0.99 vs 我们 …」对比。

⚠️ 本机交付状态：脚本未在本机运行（无 GPU / HF 不可达），报告数字待远程回填。
"""

import argparse
import gc
import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM

from evals.metrics import get_metrics
from model import get_tokenizer

logger = logging.getLogger("d_gamma_scan")

ORIG_CWD = os.getcwd()

# 测的指标（与 cfg.eval.tofu.metrics 同名）：D_KS 通道 + MU 通道
METRIC_NAMES: tuple = ("forget_quality_D", "model_utility")

# 方案 §6 机遇 3 的插值档位
DEFAULT_GAMMAS: tuple = (0.5, 0.75, 1.0, 1.25, 1.5)

# 三分支判定表原文（主方案 §6 机遇 3，照抄，禁止改写）
JUDGEMENT_TABLE_HEADER = "| 结果 | 解读 | 对搜索空间的影响 |"
JUDGEMENT_TABLE_ROWS = [
    (
        "`γ<1` 时 MU 回升而 `D_KS` 不变",
        "**崩溃主要是漂移**",
        (
            "⟹ ES 的任务不是「找方向」，而是**沿 Δ_FO 射线做步长/曲率修正** ⟹ "
            "搜索空间缩到 `S ∪ {Δ_FO}`，**K+1 维**，成本再降一个量级"
        ),
    ),
    (
        "两者同步变化",
        "真帕累托前沿",
        "ES 必须做全 K 维搜索",
    ),
    (
        "`γ>1` 外推 `D_KS` 继续降",
        "方向对、步长不够",
        "⟹ **「遗忘方向」与「步长」是可分离的两个问题**——这本身就是一个可发表的发现",
    ),
]


def _parse_args() -> argparse.Namespace:
    """CLI 参数（与 Hydra 配置互补：配置管数据/metrics，CLI 管双 ckpt 与 γ 档）。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theta-0-path",
        default=None,
        help="θ₀ 基座 ckpt（默认取 configs/experiment/eval/tofu/default.yaml 的 model）",
    )
    parser.add_argument(
        "--simnpo-path",
        required=True,
        help="θ_SimNPO 官方 ckpt（forget10，ep5 或 ep10）",
    )
    parser.add_argument(
        "--retain-logs-path",
        default=None,
        help="冻结 retain 模型在 forget 集上的 TR JSON（G0 产物）；缺失时 D_KS=None",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=list(DEFAULT_GAMMAS),
        help="插值系数档位（默认 0.5 0.75 1.0 1.25 1.5）",
    )
    parser.add_argument(
        "--output-dir", default="saves/d_gamma", help="scan.json 输出目录"
    )
    parser.add_argument("--device", default="cuda:0", help="评测设备")
    return parser.parse_args()


def _load_state_dict(path: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """CPU 上加载模型权重为 state_dict（双模型驻留做插值，1B bf16 ≈ 2.5 GB × 2）。"""
    logger.info("loading state_dict from %s", path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    sd = {k: v.detach().to(dtype=dtype) for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return sd


def _interpolate(
    sd0: dict[str, torch.Tensor],
    sd1: dict[str, torch.Tensor],
    gamma: float,
) -> dict[str, torch.Tensor]:
    """θ(γ) = θ₀ + γ·(θ_SimNPO − θ₀)，逐数值 key，fp32 精度插值后转回 bf16。

    若两 state_dict 的 key 集合不一致（异常情况），只在交集上插值并告警。
    """
    keys = set(sd0) & set(sd1)
    if len(keys) != len(sd0):
        logger.warning(
            "state_dict key mismatch: %d keys not shared; interpolating over the intersection",
            len(sd0) ^ len(sd1),
        )
    return {
        k: (sd0[k].float() + gamma * (sd1[k].float() - sd0[k].float())).to(sd0[k].dtype)
        for k in sorted(keys)
    }


def _build_model(
    base_path: str,
    sd: dict[str, torch.Tensor],
    dtype: torch.dtype,
    device: str,
) -> AutoModelForCausalLM:
    """以 θ₀ 架构实例化模型并装入插值权重。"""
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        logger.warning(
            "load_state_dict(strict=False): missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    model.to(device)
    return model


def _evaluate_gamma(
    model,
    tokenizer,
    template_args,
    cfg: DictConfig,
) -> dict[str, Any]:
    """复用仓库指标框架对单个模型测 D_KS 与 MU（cache 每 γ 独立，防串模型）。"""
    eval_cfg = cfg.eval.tofu
    cache: dict[str, Any] = {}
    results: dict[str, Any] = {}
    for name in METRIC_NAMES:
        metric_cfg = eval_cfg.metrics[name]
        metric = get_metrics(OmegaConf.create({name: metric_cfg}))[name]
        metric_kwargs = OmegaConf.to_container(metric_cfg, resolve=True)
        results[name] = metric.evaluate(
            model,
            metric_name=name,
            cache=cache,
            tokenizer=tokenizer,
            template_args=template_args,
            **metric_kwargs,
        )
    return results


def _extract_agg(results: dict[str, Any], name: str) -> float | None:
    """从指标返回值里取 agg_value；缺值/None 一律原样返回（不臆造）。"""
    value = results.get(name, {}).get("agg_value")
    if value is None:
        logger.warning("%s agg_value is None (missing reference logs?)", name)
    return value


def _judge(rows: list[dict[str, Any]]) -> str:
    """按方案 §6 机遇 3 三分支表给出初步判定（数据缺失时标注无法判定）。

    规则（初步，需人工复核）：
    - γ=0.5 时 MU 相对 γ=1.0 回升、而 D_KS 基本不变（|ΔD| ≤ 0.05）→ 漂移为主；
    - γ>1 时 D_KS 单调下降 → 方向对、步长不够；
    - 其余（两者同步变化）→ 真帕累托前沿。
    """
    finite = [r for r in rows if r.get("D_KS") is not None and r.get("MU") is not None]
    if len(finite) < 2:
        return (
            "数据缺失（retain_logs_path 未提供或指标返回 None），无法判定，待远程补跑。"
        )
    g1 = next((r for r in rows if abs(r["gamma"] - 1.0) < 1e-9), None)
    g05 = next((r for r in rows if abs(r["gamma"] - 0.5) < 1e-9), None)
    g125 = next((r for r in rows if abs(r["gamma"] - 1.25) < 1e-9), None)
    g15 = next((r for r in rows if abs(r["gamma"] - 1.5) < 1e-9), None)
    if g05 is not None and g1 is not None and g125 is not None:
        mu_recovered = g05["MU"] > g1["MU"]
        d_ks_stable = abs(g05["D_KS"] - g1["D_KS"]) <= 0.05
        if mu_recovered and d_ks_stable:
            return (
                "γ<1 时 MU 回升而 D_KS 不变 ⟹ 崩溃主要是漂移 ⟹ 搜索空间缩到 "
                "S ∪ {Δ_FO}（K+1 维），成本再降一个量级。"
            )
        d_ks_down = g125["D_KS"] < g1["D_KS"] and (
            g15 is None or g15["D_KS"] <= g125["D_KS"] + 1e-9
        )
        if d_ks_down:
            return (
                "γ>1 外推 D_KS 继续降 ⟹ 方向对、步长不够 ⟹ 「遗忘方向」与「步长」"
                "是可分离的两个问题（可发表发现）。"
            )
    return "两者同步变化 ⟹ 真帕累托前沿 ⟹ ES 必须做全 K 维搜索。"


def _write_report(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    judgement: str,
) -> None:
    """写 reports/D_gamma.md：轨迹表 + 三分支判定表原文 + Stage 5 搜索空间约束。"""
    report_path = Path("reports") / "D_gamma.md"
    lines = [
        "# D-γ 扫描报告（Stage 2 支线，方案 §6 机遇 3）",
        "",
        "零训练成本诊断：SimNPO 的效用崩溃是「漂移」（可恢复）还是「真删除」（不可逆）？",
        "",
        "## 配置",
        "",
        f"- θ₀：`{args.theta_0_path}`",
        f"- θ_SimNPO：`{args.simnpo_path}`",
        f"- retain_logs_path：`{args.retain_logs_path or '—（未提供，D_KS=None）'}`",
        f"- 设备：`{args.device}`",
        "",
        "## (γ, D_KS, MU) 轨迹",
        "",
        "| γ | D_KS | MU |",
        "|---|---|---|",
    ]
    for row in rows:
        d = "—" if row["D_KS"] is None else f"{row['D_KS']:.4f}"
        mu = "—" if row["MU"] is None else f"{row['MU']:.4f}"
        lines.append(f"| {row['gamma']} | {d} | {mu} |")
    lines += [
        "",
        "> 状态：本机交付时未运行（无 GPU / HF 不可达），数字为 `—` 占位；",
        "> 远程 2×A800 跑完 `scripts/d_gamma_scan.py` 后自动回填。",
        "",
        "## 判定（方案 §6 机遇 3，三行表原文照抄）",
        "",
        JUDGEMENT_TABLE_HEADER,
        "|---|---|---|",
    ]
    for result, meaning, impact in JUDGEMENT_TABLE_ROWS:
        lines.append(f"| {result} | {meaning} | {impact} |")
    lines += [
        "",
        f"**初步判定**：{judgement}",
        "",
        "## 对 Stage 5 搜索空间的约束",
        "",
        (
            "- 若判定为**漂移为主**：ES 的任务从「找方向」改为「沿 Δ_FO 射线做步长/曲率修正」，"
            "搜索空间缩为 `S ∪ {Δ_FO}`（**K+1 维**），种群 ES 的维度与成本再降一个量级。"
        ),
        (
            "- 若判定为**真帕累托**：ES 必须做全 K 维搜索，`accumulate_gram` 的 K 取 G1′ 定出的 "
            "`K_eff`。"
        ),
        (
            "- 若判定为**步长不足**：「遗忘方向」与「步长」解耦，`σ_t` 衰减调度只需管 `ρ_cross`"
            "（主方案 §8），方向基用 `G^{-1/2}` 白化后的 `S`。"
        ),
        "",
        (
            "> 技术先例（主方案 §6 机遇 3）：Hoy et al. 已证明 ES 与 GRPO 的解线性连通且无损失"
            "壁垒 ⟹ 权重插值是合法的分析工具。"
        ),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", report_path)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="experiment/eval/tofu/default.yaml",
)
def main(cfg: DictConfig) -> None:
    os.chdir(ORIG_CWD)  # Hydra 会 chdir 到输出目录，切回原 cwd 解析相对路径
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args()

    if args.retain_logs_path is not None:
        # OmegaConf 插值 ${eval.tofu.retain_logs_path} 会在指标配置展开时读取该值
        cfg.eval.tofu.retain_logs_path = args.retain_logs_path
    if args.theta_0_path is None:
        args.theta_0_path = cfg.model.model_args.pretrained_model_name_or_path

    tokenizer = get_tokenizer(cfg.model.tokenizer_args)
    template_args = cfg.model.template_args

    dtype = torch.bfloat16
    sd0 = _load_state_dict(args.theta_0_path, dtype)
    sd1 = _load_state_dict(args.simnpo_path, dtype)

    rows: list[dict[str, Any]] = []
    for gamma in args.gammas:
        logger.info("=== γ = %s ===", gamma)
        sd = _interpolate(sd0, sd1, gamma)
        model = _build_model(args.theta_0_path, sd, dtype, args.device)
        try:
            results = _evaluate_gamma(model, tokenizer, template_args, cfg)
        finally:
            del model, sd
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(
            {
                "gamma": gamma,
                "D_KS": _extract_agg(results, "forget_quality_D"),
                "MU": _extract_agg(results, "model_utility"),
            }
        )
        logger.info(
            "γ=%s  D_KS=%s  MU=%s",
            gamma,
            rows[-1]["D_KS"],
            rows[-1]["MU"],
        )

    judgement = _judge(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_json = {
        "task": "D-gamma scan (Stage 2, 主方案 §6 机遇 3)",
        "theta_0": args.theta_0_path,
        "simnpo": args.simnpo_path,
        "retain_logs_path": args.retain_logs_path,
        "gammas": args.gammas,
        "rows": rows,
        "judgement": judgement,
    }
    (out_dir / "scan.json").write_text(
        json.dumps(scan_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(rows, args, judgement)
    logger.info("judgement: %s", judgement)


if __name__ == "__main__":
    main()
