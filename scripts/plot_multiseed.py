from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV  = REPO_ROOT / "results" / "multiseed_random_pareto_evofolio_monthly" / "long_form.csv"
OUTPUT_DIR = INPUT_CSV.parent / "plots"

ALGO_ORDER = ["EvoFolio", "GRASP_Pareto", "Random"]
COLORS = {
    "EvoFolio":     "#d62728",
    "GRASP_Pareto": "#2ca02c",
    "Random":       "#1f77b4",
}


def plot_bars_mean_std(df: pd.DataFrame, save_path: Path) -> None:
    metrics = [
        ("total_return",      "Retorno total",           lambda v: f"{v:+.1%}"),
        ("annualised_return", "Retorno anualizado",       lambda v: f"{v:+.1%}"),
        ("sharpe",            "Sharpe ratio",             lambda v: f"{v:+.3f}"),
        ("max_drawdown",      "Max drawdown",             lambda v: f"{v:+.1%}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric, label, fmt) in zip(axes, metrics):
        means = [df[df["algorithm"] == a][metric].mean() for a in ALGO_ORDER]
        stds  = [df[df["algorithm"] == a][metric].std()  for a in ALGO_ORDER]
        colors = [COLORS[a] for a in ALGO_ORDER]

        bars = ax.bar(
            ALGO_ORDER, means, yerr=stds, capsize=8,
            color=colors, alpha=0.85, edgecolor="black", linewidth=0.8,
            error_kw=dict(ecolor="black", lw=1.5),
        )
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.grid(axis="y", alpha=0.3)

        for bar, mean, std in zip(bars, means, stds):
            y = mean + (std if mean >= 0 else -std)
            offset = 0.03 * (max(np.abs(means)) + max(stds) + 1e-9)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + (offset if mean >= 0 else -offset),
                fmt(mean),
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=10, fontweight="bold",
            )

    fig.suptitle(
        "Multi-seed (N=5) — media ± σ  |  rebalanceo mensual, "
        "return_signal=full_history",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_boxplot_by_metric(df: pd.DataFrame, save_path: Path) -> None:
    metrics = [
        ("total_return",      "Retorno total"),
        ("annualised_return", "Retorno anualizado"),
        ("sharpe",            "Sharpe ratio"),
        ("annualised_vol",    "Volatilidad anualizada"),
        ("max_drawdown",      "Max drawdown"),
        ("final_value",       "Valor final ($)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics):
        data = [df[df["algorithm"] == a][metric].values for a in ALGO_ORDER]
        bp = ax.boxplot(
            data, labels=ALGO_ORDER, patch_artist=True,
            widths=0.55, showmeans=True,
            meanprops=dict(marker="D", markeredgecolor="black",
                           markerfacecolor="white", markersize=7),
        )
        for patch, algo in zip(bp["boxes"], ALGO_ORDER):
            patch.set_facecolor(COLORS[algo])
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")

        for i, algo in enumerate(ALGO_ORDER, start=1):
            values = df[df["algorithm"] == algo][metric].values
            jitter = np.random.uniform(-0.06, 0.06, size=len(values))
            ax.scatter(np.full(len(values), i) + jitter, values,
                       color="black", s=25, zorder=3, alpha=0.75)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(
        "Distribución de métricas por semilla  |  rebalanceo mensual",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_scatter_risk_return(df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    for algo in ALGO_ORDER:
        sub = df[df["algorithm"] == algo]
        ax.scatter(
            sub["annualised_vol"], sub["annualised_return"],
            s=140, color=COLORS[algo], edgecolor="black",
            linewidth=1.0, alpha=0.85, label=f"{algo} (n={len(sub)})",
        )
        cx, cy = sub["annualised_vol"].mean(), sub["annualised_return"].mean()
        ax.scatter(cx, cy, s=380, color=COLORS[algo], edgecolor="black",
                   linewidth=1.8, marker="*", zorder=5)
        ax.annotate(
            f"  {algo}  Sharpe={sub['sharpe'].mean():+.3f}",
            xy=(cx, cy), fontsize=9, fontweight="bold",
            xytext=(6, 6), textcoords="offset points",
        )

    ax.axhline(0, color="black", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Volatilidad anualizada", fontsize=12)
    ax.set_ylabel("Retorno anualizado", fontsize=12)
    ax.set_title(
        "Risk-return por semilla (5 seeds × 3 algoritmos)  |  mensual",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best", framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_strip_final_value(df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, algo in enumerate(ALGO_ORDER):
        values = df[df["algorithm"] == algo]["final_value"].values
        seeds = df[df["algorithm"] == algo]["seed"].values
        jitter = np.random.uniform(-0.09, 0.09, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter, values,
            s=200, color=COLORS[algo], edgecolor="black", linewidth=0.8,
            alpha=0.85, zorder=3,
        )
        m = values.mean()
        ax.hlines(m, i - 0.28, i + 0.28, colors="black",
                  linewidth=2.5, zorder=4)
        ax.text(i, m + 30_000, f"mean={m:,.0f}", ha="center",
                fontsize=10, fontweight="bold")
        for x, y, s in zip(np.full(len(values), i) + jitter, values, seeds):
            ax.annotate(f" s={s}", (x, y), fontsize=8, alpha=0.7)

    ax.axhline(1_000_000, color="gray", linewidth=1.2, linestyle="--",
               label="Capital inicial ($1M)")
    ax.set_xticks(range(len(ALGO_ORDER)))
    ax.set_xticklabels(ALGO_ORDER)
    ax.set_ylabel("Valor final de cartera ($)", fontsize=12)
    ax.set_title(
        "Valor final por semilla y algoritmo  |  5 seeds, mensual",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M")
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_focused_evo_vs_pareto(
    df: pd.DataFrame,
    save_path: Path,
    signal_label: str = "full\\_history",
) -> None:
    metrics = [
        ("annualised_return", "Retorno anualizado", lambda v: f"{v:+.1%}"),
        ("sharpe",            "Sharpe ratio",       lambda v: f"{v:+.3f}"),
    ]
    focused = ["EvoFolio", "GRASP_Pareto"]
    ref     = "Random"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, (metric, label, fmt) in zip(axes, metrics):
        means = [df[df["algorithm"] == a][metric].mean() for a in focused]
        stds  = [df[df["algorithm"] == a][metric].std()  for a in focused]
        colors = [COLORS[a] for a in focused]

        bars = ax.bar(
            focused, means, yerr=stds, capsize=12,
            color=colors, alpha=0.85, edgecolor="black", linewidth=0.8,
            error_kw=dict(ecolor="black", lw=1.7),
            width=0.5,
        )

        ref_mean = df[df["algorithm"] == ref][metric].mean()
        ref_std  = df[df["algorithm"] == ref][metric].std()
        ax.axhline(ref_mean, linestyle="--", color=COLORS[ref],
                   linewidth=2.4, zorder=3,
                   label=f"Random  ref = {fmt(ref_mean)} ± {ref_std:.3f}")
        ax.axhspan(ref_mean - ref_std, ref_mean + ref_std,
                   color=COLORS[ref], alpha=0.12, zorder=1)

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=10, framealpha=0.95)

        span = max([abs(m) for m in means] + [abs(ref_mean), max(stds)]) + 1e-9
        for bar, mean, std in zip(bars, means, stds):
            y = mean + (std if mean >= 0 else -std)
            offset = 0.04 * span
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + (offset if mean >= 0 else -offset),
                fmt(mean),
                ha="center", va="bottom" if mean >= 0 else "top",
                fontsize=11, fontweight="bold",
            )

    fig.suptitle(
        f"Multi-seed (N=5): EvoFolio vs Pareto-GRASP — signal: {signal_label}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV.name}")

    print("Plotting bars_mean_std.png ...")
    plot_bars_mean_std(df, OUTPUT_DIR / "bars_mean_std.png")

    print("Plotting boxplot_by_metric.png ...")
    plot_boxplot_by_metric(df, OUTPUT_DIR / "boxplot_by_metric.png")

    print("Plotting scatter_risk_return.png ...")
    plot_scatter_risk_return(df, OUTPUT_DIR / "scatter_risk_return.png")

    print("Plotting strip_final_value.png ...")
    plot_strip_final_value(df, OUTPUT_DIR / "strip_final_value.png")

    print("Plotting focused_evo_vs_pareto.png ...")
    plot_focused_evo_vs_pareto(
        df, OUTPUT_DIR / "focused_evo_vs_pareto.png",
        signal_label="full_history",
    )

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
