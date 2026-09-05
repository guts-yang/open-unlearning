#!/usr/bin/env python3
"""Publication figure for SpecDiff on TOFU forget10 (Llama-3.2-1B-Instruct).

Reported SpecDiff point: λ=1, β=0.1, κ=0.3, τ=0.02, lr=5e-5, 10 epochs
(3 seeds). lr=2e-5 κ=0.3 is overlaid as the zero-mechanism Utility baseline.
κ=0.5 is shown on the Pareto as an ablation (Priv collapse).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
JSONL = RESULTS / "specdiff_tofu.jsonl"

# Okabe–Ito
C_ORANGE = "#E69F00"
C_VERM = "#D55E00"
C_PURPLE = "#CC79A7"
C_GRAY = "#999999"
C_BLACK = "#222222"

C_SKY = "#56B4E9"

# Self-train best-Agg per method (ou_table3.md 〇). SpecDiff bars use 3-seed mean.
HYPER_K03 = "lr5e-05_lam1_b0.1_k0.3_t0.02_ep10"
HYPER_K05 = "lr5e-05_lam1_b0.1_k0.5_t0.02_ep10"
HYPER_LR2 = "lr2e-05_lam1_b0.1_k0.3_t0.02_ep10"

BASELINES = [
    ("NPO", 0.3120, 0.6165, 0.9700, 0.5122),
    ("SimNPO", 0.3339, 0.5654, 0.8805, 0.5085),
    ("AltPO", 0.5103, 0.1914, 0.9958, 0.3664),
    ("GradDiff", 0.9381, 0.1623, 0.7714, 0.3519),
    ("RMU", 0.6043, 0.1874, 0.1655, 0.2302),
    ("UNDIAL", 0.1219, 0.2955, 0.8451, 0.2350),
]

PARETO_EXTRA = [
    ("TPO", 0.3994, 0.3086, 0.0002, 0.0007),
    ("GA", 0.9724, 0.4809, 0.0000, 0.0000),
]


def load_specdiff_rows():
    rows = []
    with JSONL.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def group_hyper(rows, hyper):
    return [row for row in rows if row["hyper"] == hyper]


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def panel_agg(ax, spec, spec_lr2=None):
    names = [r"SpecDiff" + "\n" + r"$\kappa{=}0.3$"]
    agg = [spec["Agg"][0]]
    err = [spec["Agg"][1]]
    colors = [C_ORANGE]
    if spec_lr2 is not None:
        names.append(r"SpecDiff" + "\n" + r"lr $2{\times}10^{-5}$")
        agg.append(spec_lr2["Agg"][0])
        err.append(spec_lr2["Agg"][1])
        colors.append(C_SKY)
    names += [row[0] for row in BASELINES]
    agg += [row[4] for row in BASELINES]
    err += [0.0] * len(BASELINES)
    colors += [C_GRAY] * len(BASELINES)
    x = np.arange(len(names))
    ax.bar(
        x,
        agg,
        yerr=err,
        color=colors,
        capsize=2.2,
        error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": C_BLACK},
        zorder=3,
        width=0.72,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel(r"Agg $=$ HM(Mem, Priv, Utility)")
    ax.set_ylim(0.0, 0.72)
    ax.set_title("(a) Aggregate score (higher is better)")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#cccccc", zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(
        f"{agg[0]:.3f}",
        (0, agg[0] + err[0] + 0.018),
        ha="center",
        va="bottom",
        fontsize=8,
        color=C_VERM,
        fontweight="bold",
    )
    if spec_lr2 is not None:
        ax.annotate(
            f"{agg[1]:.3f}",
            (1, agg[1] + err[1] + 0.018),
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_SKY,
            fontweight="bold",
        )


def panel_pareto(ax, spec, rows):
    offsets = {
        "NPO": (0.00, 0.045),
        "SimNPO": (0.06, -0.055),
        "AltPO": (0.00, 0.045),
        "GradDiff": (0.04, 0.04),
        "RMU": (0.00, -0.055),
        "UNDIAL": (0.00, 0.045),
        "TPO": (0.05, 0.04),
        "GA": (0.06, 0.04),
    }
    for name, mem, priv, util, agg in BASELINES + PARETO_EXTRA:
        ax.scatter(
            util,
            priv,
            s=40 + 180 * mem,
            color=C_GRAY,
            edgecolors=C_BLACK,
            linewidths=0.4,
            alpha=0.75,
            zorder=3,
        )
        dx, dy = offsets.get(name, (0.0, 0.04))
        ax.annotate(name, (util + dx, priv + dy), ha="center", fontsize=6.5, color="#555555")

    ax.scatter(
        spec["Utility"][0],
        spec["Priv"][0],
        s=40 + 180 * spec["Mem"][0],
        color=C_ORANGE,
        edgecolors=C_BLACK,
        linewidths=0.7,
        zorder=5,
        marker="*",
    )
    ax.errorbar(
        spec["Utility"][0],
        spec["Priv"][0],
        xerr=spec["Utility"][1],
        yerr=spec["Priv"][1],
        fmt="none",
        ecolor=C_VERM,
        elinewidth=0.8,
        capsize=2,
        zorder=4,
    )
    ax.annotate(
        r"SpecDiff ($\kappa{=}0.3$, lr $5{\times}10^{-5}$)",
        (spec["Utility"][0] - 0.04, spec["Priv"][0] + 0.07),
        ha="center",
        fontsize=7.5,
        color=C_VERM,
        fontweight="bold",
    )

    lr2 = group_hyper(rows, HYPER_LR2)
    if lr2:
        u, ue = mean_std([r["Utility"] for r in lr2])
        p, pe = mean_std([r["Priv"] for r in lr2])
        m, _ = mean_std([r["Mem"] for r in lr2])
        ax.scatter(
            u,
            p,
            s=40 + 180 * m,
            color=C_SKY,
            edgecolors=C_BLACK,
            linewidths=0.7,
            zorder=5,
            marker="D",
        )
        ax.errorbar(
            u,
            p,
            xerr=ue,
            yerr=pe,
            fmt="none",
            ecolor=C_SKY,
            elinewidth=0.8,
            capsize=2,
            zorder=4,
        )
        ax.annotate(
            r"lr $2{\times}10^{-5}$ (Utility rescue)",
            (u - 0.02, p + 0.06),
            ha="center",
            fontsize=6.5,
            color=C_SKY,
            fontweight="bold",
        )

    # κ=0.5 mean: high Mem, collapsed Priv
    k5 = group_hyper(rows, HYPER_K05)
    if k5:
        u, _ = mean_std([r["Utility"] for r in k5])
        p, _ = mean_std([r["Priv"] for r in k5])
        m, _ = mean_std([r["Mem"] for r in k5])
        ax.scatter(
            u,
            p,
            s=40 + 180 * m,
            facecolors="none",
            edgecolors=C_PURPLE,
            linewidths=1.1,
            zorder=5,
            marker="o",
        )
        ax.annotate(
            r"$\kappa{=}0.5$ (Priv collapse)",
            (u + 0.02, p - 0.02),
            fontsize=6.5,
            color=C_PURPLE,
        )

    ax.set_xlabel("Utility")
    ax.set_ylabel("Priv")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 0.95)
    ax.set_title(r"(b) Privacy–utility Pareto (area $\propto$ Mem)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=C_ORANGE,
            markeredgecolor=C_BLACK,
            markersize=10,
            label=r"SpecDiff $\kappa{=}0.3$ lr $5{\times}10^{-5}$",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=C_SKY,
            markeredgecolor=C_BLACK,
            markersize=7,
            label=r"SpecDiff lr $2{\times}10^{-5}$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_GRAY,
            markeredgecolor=C_BLACK,
            markersize=7,
            label="Other methods (best Agg)",
        ),
    ]
    ax.legend(handles=legend, frameon=False, loc="lower left")


def main():
    setup_style()
    rows = load_specdiff_rows()
    reported = group_hyper(rows, HYPER_K03)
    spec = {
        key: mean_std([row[key] for row in reported])
        for key in ("Mem", "Priv", "Utility", "Agg")
    }
    lr2_rows = group_hyper(rows, HYPER_LR2)
    spec_lr2 = None
    if lr2_rows:
        spec_lr2 = {
            key: mean_std([row[key] for row in lr2_rows])
            for key in ("Mem", "Priv", "Utility", "Agg")
        }

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.15), layout="constrained")
    panel_agg(axes[0], spec, spec_lr2)
    panel_pareto(axes[1], spec, rows)

    fig.suptitle(
        "SpecDiff on TOFU forget10  ·  Llama-3.2-1B-Instruct  ·  self-trained",
        fontsize=10,
    )
    out_png = RESULTS / "specdiff_fig_main.png"
    out_pdf = RESULTS / "specdiff_fig_main.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    print(
        "reported κ=0.3 n="
        f"{len(reported)}  Mem={spec['Mem'][0]:.3f}±{spec['Mem'][1]:.3f}  "
        f"Priv={spec['Priv'][0]:.3f}±{spec['Priv'][1]:.3f}  "
        f"Util={spec['Utility'][0]:.3f}±{spec['Utility'][1]:.3f}  "
        f"Agg={spec['Agg'][0]:.3f}±{spec['Agg'][1]:.3f}"
    )
    if spec_lr2 is not None:
        print(
            "lr=2e-5 n="
            f"{len(lr2_rows)}  Mem={spec_lr2['Mem'][0]:.3f}±{spec_lr2['Mem'][1]:.3f}  "
            f"Priv={spec_lr2['Priv'][0]:.3f}±{spec_lr2['Priv'][1]:.3f}  "
            f"Util={spec_lr2['Utility'][0]:.3f}±{spec_lr2['Utility'][1]:.3f}  "
            f"Agg={spec_lr2['Agg'][0]:.3f}±{spec_lr2['Agg'][1]:.3f}"
        )


if __name__ == "__main__":
    main()
