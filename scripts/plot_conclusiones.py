from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = REPO_ROOT / "results" / "conclusiones"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"EvoFolio": "#d62728", "GRASP_Pareto": "#2ca02c"}


def load_metrics(csv_path: Path, algos: list[str]) -> dict:
    df = pd.read_csv(csv_path)
    out = {}
    for a in algos:
        sub = df[df["algorithm"] == a]
        out[a] = {
            "sharpe_mean":       sub["sharpe"].mean(),
            "sharpe_std":        sub["sharpe"].std(),
            "annret_mean":       sub["annualised_return"].mean(),
            "annret_std":        sub["annualised_return"].std(),
            "annvol_mean":       sub["annualised_vol"].mean(),
            "annvol_std":        sub["annualised_vol"].std(),
            "maxdd_mean":        sub["max_drawdown"].mean(),
            "maxdd_std":         sub["max_drawdown"].std(),
        }
    return out


def plot_triangulation(save_path: Path) -> None:
    scenarios = [
        ("full_history\nmensual",     "multiseed_random_pareto_evofolio_monthly"),
        ("EWMA (h=12)\nmensual",      "multiseed_ewma_monthly"),
        ("BL/EWMA-views\ntrimestral", "multiseed_bl_views_ewma_quarterly"),
    ]

    data = {"EvoFolio": [], "GRASP_Pareto": []}
    stds = {"EvoFolio": [], "GRASP_Pareto": []}

    for label, folder in scenarios:
        csv_path = REPO_ROOT / "results" / folder / "long_form.csv"
        m = load_metrics(csv_path, ["EvoFolio", "GRASP_Pareto"])
        for a in ("EvoFolio", "GRASP_Pareto"):
            data[a].append(m[a]["sharpe_mean"])
            stds[a].append(m[a]["sharpe_std"])

    labels = [s[0] for s in scenarios]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width/2, data["EvoFolio"], width,
                yerr=stds["EvoFolio"], capsize=8,
                color=COLORS["EvoFolio"], alpha=0.87, edgecolor="black",
                linewidth=0.7, label="EvoFolio",
                error_kw=dict(ecolor="black", lw=1.5))
    b2 = ax.bar(x + width/2, data["GRASP_Pareto"], width,
                yerr=stds["GRASP_Pareto"], capsize=8,
                color=COLORS["GRASP_Pareto"], alpha=0.87, edgecolor="black",
                linewidth=0.7, label="Pareto-GRASP",
                error_kw=dict(ecolor="black", lw=1.5))

    all_top = [data[a][i] + stds[a][i] for a in data for i in range(len(labels))]
    all_bot = [data[a][i] - stds[a][i] for a in data for i in range(len(labels))]
    data_max = max(all_top + [0.0])
    data_min = min(all_bot + [0.0])
    data_span = max(data_max - data_min, 1e-9)

    ax.set_ylim(data_min - 0.20 * data_span, data_max + 0.25 * data_span)
    label_offset = 0.05 * data_span

    for bar, v, s in zip(b1, data["EvoFolio"], stds["EvoFolio"]):
        if v >= 0:
            y_tip = v + s; y_text = y_tip + label_offset; va = "bottom"
        else:
            y_tip = v - s; y_text = y_tip - label_offset; va = "top"
        ax.text(bar.get_x() + bar.get_width()/2, y_text,
                f"{v:+.3f}", ha="center", va=va,
                fontsize=10, fontweight="bold")

    for bar, v, s in zip(b2, data["GRASP_Pareto"], stds["GRASP_Pareto"]):
        if v >= 0:
            y_tip = v + s; y_text = y_tip + label_offset; va = "bottom"
        else:
            y_tip = v - s; y_text = y_tip - label_offset; va = "top"
        ax.text(bar.get_x() + bar.get_width()/2, y_text,
                f"{v:+.3f}", ha="center", va=va,
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Sharpe ratio anualizado", fontsize=12)
    ax.set_title(
        "Triangulación de la calidad de señal — Sharpe multi-seed (N=5)\n"
        "EvoFolio vs Pareto-GRASP en tres escenarios",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def plot_bl_focused(save_path: Path) -> None:
    csv_path = REPO_ROOT / "results" / "multiseed_bl_views_ewma_quarterly" / "long_form.csv"
    m = load_metrics(csv_path, ["EvoFolio", "GRASP_Pareto"])

    metrics = [
        ("annret",  "Retorno anualizado",    lambda v: f"{v:+.1%}"),
        ("annvol",  "Volatilidad anualizada", lambda v: f"{v:.1%}"),
        ("sharpe",  "Sharpe ratio",           lambda v: f"{v:+.3f}"),
        ("maxdd",   "Max drawdown",           lambda v: f"{v:+.1%}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    algos = ["EvoFolio", "GRASP_Pareto"]
    colors = [COLORS[a] for a in algos]

    for ax, (key, title, fmt) in zip(axes, metrics):
        means = [m[a][f"{key}_mean"] for a in algos]
        stds  = [m[a][f"{key}_std"]  for a in algos]
        bars = ax.bar(algos, means, yerr=stds, capsize=10,
                      color=colors, alpha=0.87, edgecolor="black",
                      linewidth=0.8, width=0.55,
                      error_kw=dict(ecolor="black", lw=1.5))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.grid(axis="y", alpha=0.3)

        top_extents = [means[i] + stds[i] for i in range(len(means))]
        bot_extents = [means[i] - stds[i] for i in range(len(means))]
        data_max = max(top_extents + [0.0])
        data_min = min(bot_extents + [0.0])
        data_span = max(data_max - data_min, 1e-9)

        padding = 0.30 * data_span
        ax.set_ylim(data_min - padding, data_max + padding)

        label_offset = 0.08 * data_span
        for bar, v, s in zip(bars, means, stds):
            if v >= 0:
                y_tip = v + s
                y_text = y_tip + label_offset
                va = "bottom"
            else:
                y_tip = v - s
                y_text = y_tip - label_offset
                va = "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y_text, fmt(v),
                    ha="center", va=va,
                    fontsize=11, fontweight="bold")

    fig.suptitle(
        "Multi-seed (N=5) — BL/EWMA-views trimestral\n"
        "EvoFolio vs Pareto-GRASP — media $\\pm$ desviación típica",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    print("Plotting triangulation_signal_quality.png ...")
    plot_triangulation(OUTPUT_DIR / "triangulation_signal_quality.png")
    print("Plotting bl_quarterly_evo_vs_pareto.png ...")
    plot_bl_focused(OUTPUT_DIR / "bl_quarterly_evo_vs_pareto.png")
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
