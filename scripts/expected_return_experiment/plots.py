from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_scatter_grid(
    dfs_by_estimator: dict,
    metrics_by_estimator: dict,
    freq_label: str,
    save_path: str | Path,
) -> None:
    names = list(dfs_by_estimator.keys())
    n = len(names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    axes = np.array(axes).reshape(-1)

    all_p = np.concatenate([dfs_by_estimator[k]["predicted"].values for k in names])
    all_r = np.concatenate([dfs_by_estimator[k]["realized"].values for k in names])
    lim = float(max(np.abs(all_p).max(), np.abs(all_r).max())) * 1.05

    for ax, name in zip(axes, names):
        df = dfs_by_estimator[name]
        m = metrics_by_estimator[name]
        ax.scatter(df["predicted"], df["realized"], alpha=0.35, s=15,
                   edgecolor="none", color="steelblue")
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, linewidth=1,
                label="perfect prediction")
        ax.axhline(0, color="gray", alpha=0.3, linewidth=0.5)
        ax.axvline(0, color="gray", alpha=0.3, linewidth=0.5)
        ax.set_title(
            f'{name}\n'
            f'IC={m["ic"]:+.3f}  RankIC={m["rank_ic"]:+.3f}  '
            f'DirAcc={m["dir_acc"]:.1%}',
            fontsize=10,
        )
        ax.set_xlabel("Predicted cumulative log return")
        ax.set_ylabel("Realised cumulative log return")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", "box")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Predicted vs realised return — {freq_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_metrics_bars(
    metrics_by_estimator: dict,
    freq_label: str,
    save_path: str | Path,
) -> None:
    estimators = list(metrics_by_estimator.keys())
    metric_names = ["rmse", "mae", "ic", "rank_ic", "dir_acc", "r2"]
    metric_labels = [
        "RMSE  (↓ better)",
        "MAE  (↓ better)",
        "Information Coeff.  (↑ better)",
        "Rank IC  (↑ better)",
        "Directional accuracy  (↑ better)",
        "R²  (↑ better)",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(estimators)))

    for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[i]
        values = [metrics_by_estimator[e][metric] for e in estimators]
        bars = ax.bar(range(len(estimators)), values, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(estimators)))
        ax.set_xticklabels(estimators, rotation=25, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.grid(axis="y", alpha=0.3)

        vmin, vmax = min(values), max(values)
        span = max(abs(vmax - vmin), 1e-9)
        for bar, v in zip(bars, values):
            offset = 0.02 * span
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + (offset if v >= 0 else -offset),
                f"{v:.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )

    fig.suptitle(f"Prediction-quality metrics — {freq_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_cross_sectional_ic(
    ic_by_estimator: dict,
    freq_label: str,
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ic_by_estimator)))

    for color, (name, ic_series) in zip(colors, ic_by_estimator.items()):
        mean_ic = float(np.nanmean(ic_series.values))
        std_ic = float(np.nanstd(ic_series.values))
        ax.plot(ic_series.index, ic_series.values,
                marker="o", markersize=4.5, linewidth=1.5, color=color,
                label=f"{name}   mean={mean_ic:+.3f}  σ={std_ic:.3f}",
                alpha=0.9)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Cross-sectional IC over time — {freq_label}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Rebalancing date")
    ax.set_ylabel("Pearson IC across the 47 assets")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_per_asset_series(
    dfs_by_estimator: dict,
    sample_assets: list[str],
    freq_label: str,
    save_path: str | Path,
) -> None:
    n_assets = len(sample_assets)
    fig, axes = plt.subplots(n_assets, 1,
                             figsize=(11, 2.8 * n_assets),
                             sharex=True)
    if n_assets == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(dfs_by_estimator)))

    for ax, asset in zip(axes, sample_assets):
        first_df = next(iter(dfs_by_estimator.values()))
        df_true = first_df[first_df["asset"] == asset]
        ax.plot(df_true["date"], df_true["realized"],
                "-o", color="black", markersize=4.5,
                label="Realised", linewidth=1.8)

        for color, (name, df) in zip(colors, dfs_by_estimator.items()):
            df_a = df[df["asset"] == asset]
            ax.plot(df_a["date"], df_a["predicted"], "--",
                    color=color, alpha=0.85, linewidth=1.3,
                    label=f"Pred: {name}")

        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_title(asset, fontsize=11, fontweight="bold")
        ax.set_ylabel("Cum. log return")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Rebalancing date")
    fig.suptitle(f"Predicted vs realised per asset — {freq_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_price_with_predictions(
    instance,
    dfs_by_estimator: dict,
    sample_assets: list[str],
    freq_label: str,
    save_path: str | Path,
) -> None:
    n_assets = len(sample_assets)
    fig, axes = plt.subplots(n_assets, 1,
                             figsize=(12, 3.0 * n_assets),
                             sharex=True)
    if n_assets == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(dfs_by_estimator)))
    prices = instance.prices
    assets_list = list(instance.assets)

    for ax, asset in zip(axes, sample_assets):
        if asset not in assets_list:
            ax.set_title(f"{asset} (not in universe)")
            continue
        j = assets_list.index(asset)
        p_series = prices.iloc[:, j]
        ax.plot(p_series.index, p_series.values,
                "-", color="black", linewidth=1.5, label="Actual price",
                alpha=0.9)

        for color, (name, df) in zip(colors, dfs_by_estimator.items()):
            df_a = df[df["asset"] == asset]
            projected = []
            projected_dates = []
            for _, row in df_a.iterrows():
                date = row["date"]
                if date not in p_series.index:
                    continue
                p_now = p_series.loc[date]
                projected_price = p_now * np.exp(row["predicted"])
                idx_now = p_series.index.get_loc(date)
                idx_next = min(idx_now + row["period"], len(p_series) - 1)
                projected_dates.append(p_series.index[idx_next])
                projected.append(projected_price)
            ax.plot(projected_dates, projected, "o", color=color,
                    markersize=4, alpha=0.75, label=f"Pred: {name}")

        ax.set_title(asset, fontsize=11, fontweight="bold")
        ax.set_ylabel("Price")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Price trajectory with projected endpoints — {freq_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_sensitivity_grid(
    sweep_results_by_freq: dict,
    sweep_label: str,
    param_name: str,
    save_path: str | Path,
    logx: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    metric_names = ["ic", "rank_ic", "dir_acc", "rmse"]
    metric_labels = [
        "Information Coeff. (↑ better)",
        "Rank IC (↑ better)",
        "Directional accuracy (↑ better)",
        "RMSE (↓ better)",
    ]

    colors = {"Monthly (4-week horizon)":   "steelblue",
              "Quarterly (12-week horizon)": "darkorange"}

    for ax, metric, label in zip(axes, metric_names, metric_labels):
        for freq_label, df in sweep_results_by_freq.items():
            color = colors.get(freq_label, "gray")
            ax.plot(df["param_value"], df[metric], marker="o",
                    markersize=6, linewidth=1.7, color=color,
                    label=freq_label, alpha=0.9)

            if metric in ("ic", "rank_ic", "dir_acc"):
                best_idx = df[metric].idxmax()
            else:
                best_idx = df[metric].idxmin()
            best_val = df[metric].iloc[best_idx]
            best_x = df["param_value"].iloc[best_idx]
            ax.scatter([best_x], [best_val], s=140, color=color,
                       marker="*", edgecolor="black", zorder=5,
                       label=f"  best @ {param_name}={best_x}")

        if metric in ("ic", "rank_ic"):
            ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        if metric == "dir_acc":
            ax.axhline(0.5, color="black", linewidth=0.7,
                       alpha=0.5, linestyle="--", label="random (50%)")

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel(param_name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        if logx:
            ax.set_xscale("log")

    fig.suptitle(f"Hyperparameter sensitivity: {sweep_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
