import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm


COLORS  = {"Greedy": "steelblue", "GRASP": "darkorange",
           "Random": "seagreen",  "1/N":   "mediumpurple",
           "EvoFolio": "crimson", "GRASP_Pareto": "teal"}
MARKERS = {"Greedy": "o",          "GRASP": "s",
           "Random": "^",          "1/N":   "D",
           "EvoFolio": "P",        "GRASP_Pareto": "X"}


def _finalize(fig, save_path: str | None = None, show: bool = True):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_risk_return(solutions, overlay_solutions=None,
                     save_path: str | None = None, show: bool = True):
    if len(solutions) == 0:
        raise ValueError("solutions cannot be empty.")

    risks   = np.array([s.risk for s in solutions],            dtype=float)
    returns = np.array([s.expected_return for s in solutions], dtype=float)
    sharpes = np.array([s.sharpe for s in solutions],          dtype=float)

    best_idx = int(np.argmax(sharpes))

    fig, ax = plt.subplots(figsize=(9, 6), dpi=130)
    scatter = ax.scatter(
        risks, returns, c=sharpes, cmap="viridis",
        s=60, alpha=0.78, edgecolors="black", linewidths=0.35,
        label="Random portfolios",
    )
    ax.scatter(
        risks[best_idx], returns[best_idx],
        s=220, marker="*", color="red", edgecolors="black", linewidths=0.8,
        label=f"Best random (Sharpe={sharpes[best_idx]:.3f})", zorder=5,
    )
    if overlay_solutions:
        for sol, label, color, marker in overlay_solutions:
            ax.scatter(
                sol.risk, sol.expected_return,
                s=280, marker=marker, color=color, edgecolors="black", linewidths=0.8,
                label=f"{label} (Sharpe={sol.sharpe:.3f})", zorder=6,
            )

    ax.set_title("Risk-return space of feasible portfolios",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Risk", fontsize=11)
    ax.set_ylabel("Expected return", fontsize=11)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Sharpe ratio")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _finalize(fig, save_path=save_path, show=show)


def plot_weights(solution, assets, instance=None, t=None, daily_returns=None,
                 save_path: str | None = None, show: bool = True):
    weights = np.asarray(solution.weights, dtype=float)
    assets  = np.asarray(assets)

    if len(weights) != len(assets):
        raise ValueError("solution.weights and assets must have the same length.")

    if daily_returns is not None:
        if hasattr(daily_returns, "reindex"):
            returns_all = daily_returns.reindex(assets).values.astype(float)
        else:
            returns_all = np.asarray(daily_returns, dtype=float)
    elif instance is not None and t is not None:
        if t < 0 or t >= len(instance.returns):
            raise ValueError("t is out of bounds for instance.returns.")
        returns_row = instance.returns.iloc[t]
        returns_all = returns_row.reindex(assets).values.astype(float)
    else:
        raise ValueError("Provide either daily_returns or both instance and t.")

    returns_all = np.nan_to_num(returns_all, nan=0.0)
    realized_portfolio_return = float(np.sum(weights * returns_all))

    selected = weights > 1e-8
    selected_assets  = assets[selected]
    selected_weights = weights[selected]
    selected_returns = returns_all[selected]

    if len(selected_assets) == 0:
        raise ValueError("The solution has no selected assets.")

    order = np.argsort(selected_weights)
    selected_assets  = selected_assets[order]
    selected_weights = selected_weights[order]
    selected_returns = selected_returns[order]

    vmax = float(np.nanmax(np.abs(selected_returns)))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cm.get_cmap("RdYlGn")
    colors = cmap(norm(selected_returns))

    fig_height = max(5, 0.45 * len(selected_assets))
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=130)

    bars = ax.barh(
        selected_assets, selected_weights, color=colors,
        alpha=0.90, edgecolor="black", linewidth=0.4,
    )

    if instance is not None and t is not None:
        day_label = f" | t={t}: {str(instance.returns.index[t])}"
    elif t is not None:
        day_label = f" | t={t}"
    else:
        day_label = ""

    ax.set_title(
        f"Portfolio allocation\n"
        f"Expected return = {solution.expected_return:.2%} | "
        f"Realized return = {realized_portfolio_return:.2%} | "
        f"Risk = {solution.risk:.2%} | Sharpe = {solution.sharpe:.2f}"
        f"{day_label}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Portfolio weight", fontsize=11)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.25)

    ax.bar_label(bars, labels=[f"{w:.1%}" for w in selected_weights],
                 padding=4, fontsize=8)
    for bar, ret in zip(bars, selected_returns):
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f"{ret:.2%}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="black")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax); cbar.set_label("Period asset return")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _finalize(fig, save_path=save_path, show=show)


def plot_market_day_grid(instance, t, save_path: str | None = None, show: bool = True):
    if t < 0 or t >= len(instance.returns):
        raise ValueError("t is out of bounds for instance.returns")

    returns_row = instance.returns.iloc[t]
    assets = getattr(instance, "assets", list(instance.returns.columns))
    returns = returns_row.values.astype(float)
    n = len(returns)
    if n == 0:
        raise ValueError("No assets found.")

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    xs, ys, labels, vals = [], [], [], []
    for i in range(n):
        xs.append(i % ncols)
        ys.append(nrows - 1 - i // ncols)
        labels.append(assets[i])
        vals.append(returns[i])

    xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)
    abs_vals = np.abs(vals)
    sizes = (np.full_like(abs_vals, 1800.0) if abs_vals.max() == 0
             else 800 + 3200 * (abs_vals / abs_vals.max()))

    vmax = float(np.max(np.abs(vals))) or 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cm.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=130)
    scatter = ax.scatter(xs, ys, s=sizes, c=vals, cmap=cmap, norm=norm,
                         marker="s", edgecolors="black", linewidths=0.8, alpha=0.95)
    for x, y, label, value in zip(xs, ys, labels, vals):
        ax.text(x, y + 0.10, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="black")
        ax.text(x, y - 0.18, f"{value:.2%}", ha="center", va="center",
                fontsize=7, color="black")

    title = f"Market grid - t={t}"
    if len(instance.returns.index) > t:
        title = f"Market grid - t={t} ({instance.returns.index[t]})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.8, ncols - 0.2); ax.set_ylim(-0.8, nrows - 0.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.9)
    cbar.set_label("Period return")
    fig.tight_layout()
    _finalize(fig, save_path=save_path, show=show)


def plot_target_period(results: dict, cfg,
                       save_path: str | None = None, show: bool = True):
    target_idx = cfg.target_period_index
    if target_idx is None:
        return

    first_result = next(iter(results.values()))
    if target_idx >= len(first_result.periods):
        print(f"⚠ target_period_index={target_idx} out of range "
              f"(only {len(first_result.periods)} periods). Skipping scatter.")
        return

    period_info = first_result.periods[target_idx]
    date_label = str(period_info.date)[:10]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)

    for name, result in results.items():
        candidates = result.target_candidates
        if not candidates:
            continue

        color  = COLORS.get(name, "gray")
        marker = MARKERS.get(name, "o")

        risks   = np.array([c.risk for c in candidates],            dtype=float)
        returns = np.array([c.expected_return for c in candidates], dtype=float)

        if name == "Greedy":
            ax.scatter(risks, returns, s=260, marker="D", color=color,
                       edgecolors="black", linewidths=0.8,
                       label=f"{name} (selected)", zorder=6)
            continue

        ax.scatter(risks, returns, s=35, marker=marker, color=color,
                   alpha=0.30, edgecolors="black", linewidths=0.25,
                   label=f"{name} candidates ({len(candidates)})")

        sel = result.periods[target_idx]
        sel_risk = next(
            (c.risk for c in candidates
             if np.allclose(c.weights, sel.weights, atol=1e-8)),
            None,
        )
        sel_ret = next(
            (c.expected_return for c in candidates
             if np.allclose(c.weights, sel.weights, atol=1e-8)),
            None,
        )
        if sel_risk is None:
            best = max(candidates, key=lambda c: c.sharpe)
            sel_risk, sel_ret = best.risk, best.expected_return

        ax.scatter(sel_risk, sel_ret, s=300, marker="*", color=color,
                   edgecolors="black", linewidths=0.9,
                   label=f"{name} selected", zorder=7)

    banner_lines = []
    for name, result in results.items():
        if target_idx < len(result.periods):
            r = result.periods[target_idx].realized_log_return
            banner_lines.append(f"{name}: realised={r:.2%}")

    ax.set_title(
        "Candidate portfolios at rebalancing date\n"
        f"{date_label}  |  period index = {target_idx}\n"
        + "  •  ".join(banner_lines),
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Risk (σ)")
    ax.set_ylabel("Expected return (μ)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=9, loc="best")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _finalize(fig, save_path=save_path, show=show)


def plot_evofolio_evolution(results: dict, instance, cfg, period: int,
                            save_path: str | None = None, show: bool = True):
    if not results:
        return

    any_result = next(iter(results.values()))
    if len(any_result.periods) == 0:
        return
    reb_dates = pd.to_datetime(any_result.dates)
    start_date = reb_dates[0]
    end_date = reb_dates[-1]

    fig = plt.figure(figsize=(13, 8), dpi=130)
    plt.style.use("ggplot")

    best_name, best_period = None, None
    best_value = -np.inf
    for name, result in results.items():
        for p in result.periods:
            if p.portfolio_value > best_value:
                best_value = p.portfolio_value
                best_name, best_period = name, p
    if best_period is not None:
        sel = np.where(best_period.shares > 0)[0]
        composition = ", ".join(
            f"{instance.assets[i]}:{int(best_period.shares[i])}" for i in sel
        )
        fig.suptitle(
            f"Best Portfolio  ({best_name})  "
            f"{str(best_period.date)[:10]}  →  value = {best_value:,.0f}\n"
            f"{composition}",
            fontsize=11, fontweight="bold",
        )

    yplt = plt.subplot2grid((3, 3), loc=(0, 0), colspan=3)
    rplt = plt.subplot2grid((3, 3), loc=(1, 0), colspan=3)
    gplt = plt.subplot2grid((3, 3), loc=(2, 0), colspan=3)

    for name, result in results.items():
        color = COLORS.get(name, "gray")

        trace_chunks = [p.trace for p in result.periods
                        if p.trace is not None and len(p.trace) > 0]
        if not trace_chunks:
            continue
        full_trace = np.concatenate(trace_chunks, axis=0)
        if full_trace.size == 0:
            continue

        step_dates = pd.date_range(start_date, end_date, periods=len(full_trace))

        risks_series = full_trace[:, 0]
        rets_series = full_trace[:, 1]

        yplt.plot(step_dates, rets_series, color=color, linewidth=1.0, alpha=0.85,
                  label=f"{name}  (yield avg = {np.nanmean(rets_series):.4f})")
        rplt.plot(step_dates, risks_series, color=color, linewidth=1.0, alpha=0.85,
                  label=f"{name}  (risk avg = {np.nanmean(risks_series):.4f})")

        marker = MARKERS.get(name, "o")
        gplt.plot(reb_dates, result.portfolio_values, color=color, linewidth=1.8,
                  marker=marker, markersize=5,
                  label=f"{name}  (final = {result.portfolio_values[-1]:,.0f})")

    yplt.set_title("Yield (avg per search step)", fontsize=10)
    yplt.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    yplt.legend(frameon=False, fontsize=9, loc="best")
    yplt.grid(True, alpha=0.25)

    rplt.set_title("Risk (avg per search step)", fontsize=10)
    rplt.legend(frameon=False, fontsize=9, loc="best")
    rplt.grid(True, alpha=0.25)

    gplt.set_title("Portfolio value", fontsize=10)
    gplt.axhline(cfg.budget, color="gray", linewidth=0.7, linestyle="--",
                 alpha=0.5, label=f"Initial = {cfg.budget:,.0f}")
    gplt.legend(frameon=False, fontsize=9, loc="best")
    gplt.grid(True, alpha=0.25)
    gplt.set_xlabel(
        f"Date  |  rebalancing = {cfg.rebalancing} "
        f"(period = {period} {instance.frequency.lower()} rows)",
        fontsize=9,
    )

    for ax in (yplt, rplt, gplt):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout(h_pad=-0.5)
    plt.gcf().autofmt_xdate()
    _finalize(fig, save_path=save_path, show=show)


def plot_evolution(results: dict, rebalancing_label: str = "period",
                   rolling_window: int = 4, initial_value: float = 1.0,
                   save_path: str | None = None, show: bool = True):
    """Plot cumulative value, per-period returns and rolling metrics for every algorithm.

    In: results (dict[str, BacktestResult]).
    Out: None (shows and/or saves the figure).
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), dpi=130, sharex=True)
    fig.suptitle("Portfolio Evolution — Algorithm Comparison",
                 fontsize=15, fontweight="bold")

    for name, result in results.items():
        color  = COLORS.get(name, "gray")
        marker = MARKERS.get(name, "o")
        dates    = result.dates
        log_rets = result.realized_log_returns

        if len(dates) == 0:
            continue

        cum_log = np.cumsum(log_rets)
        cum_val = initial_value * np.exp(cum_log)

        period_length = (dates[1] - dates[0]) if len(dates) > 1 else pd.Timedelta(days=7)
        all_dates = [dates[0] - period_length] + list(dates)
        all_vals  = np.concatenate([[initial_value], cum_val])

        axes[0].plot(all_dates, all_vals, color=color, linewidth=2.0,
                     label=f"{name}  (final: {all_vals[-1]:.3f}×)", marker=marker,
                     markersize=4, markevery=max(1, len(all_dates) // 12))
        axes[0].fill_between(all_dates, initial_value, all_vals, alpha=0.08, color=color)

        axes[1].plot(dates, log_rets, color=color, linewidth=1.4,
                     marker=marker, markersize=4, alpha=0.85, label=name)
        axes[1].fill_between(dates, log_rets, 0, where=(log_rets >= 0),
                             color=color, alpha=0.15)
        axes[1].fill_between(dates, log_rets, 0, where=(log_rets < 0),
                             color="red", alpha=0.08)

        rolling_sh = np.full(len(log_rets), np.nan)
        for j in range(len(log_rets)):
            w = log_rets[max(0, j - rolling_window + 1): j + 1]
            if len(w) >= 2 and w.std() > 1e-10:
                rolling_sh[j] = w.mean() / w.std()
        axes[2].plot(dates, rolling_sh, color=color, linewidth=1.6,
                     marker=marker, markersize=4, label=name)

    axes[0].axhline(initial_value, color="gray", linewidth=0.8,
                    linestyle="--", alpha=0.6)
    axes[0].set_ylabel("Portfolio value", fontsize=10)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(True, alpha=0.25)

    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[1].set_ylabel(f"Realised log return\n({rebalancing_label})", fontsize=10)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    axes[1].grid(True, alpha=0.25)

    axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[2].set_ylabel(f"Rolling Sharpe\n({rolling_window} periods)", fontsize=10)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=9, loc="upper left")

    tick_interval = 1 if rebalancing_label == "quarterly" else 2
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=tick_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[2].set_xlabel("Rebalancing date", fontsize=10)
    fig.tight_layout()
    _finalize(fig, save_path=save_path, show=show)


def plot_sharpe_evolution(results: dict, cfg, periods_per_year: float,
                          rolling_window: int = 4,
                          save_path: str | None = None,
                          show: bool = True):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), dpi=130, sharex=True)
    fig.suptitle("Sharpe Ratio Evolution — Algorithm Comparison",
                 fontsize=15, fontweight="bold")

    sqrt_ppy = float(np.sqrt(periods_per_year))

    for name, result in results.items():
        color  = COLORS.get(name, "gray")
        marker = MARKERS.get(name, "o")
        dates    = result.dates
        log_rets = result.realized_log_returns
        exp_sh   = result.expected_sharpes

        if len(dates) == 0:
            continue

        axes[0].plot(dates, exp_sh, color=color, linewidth=1.6,
                     marker=marker, markersize=4, alpha=0.9,
                     label=f"{name}  (mean = {np.nanmean(exp_sh):.3f})")

        rolling_sh = np.full(len(log_rets), np.nan)
        for j in range(len(log_rets)):
            w = log_rets[max(0, j - rolling_window + 1): j + 1]
            if len(w) >= 2 and w.std() > 1e-10:
                rolling_sh[j] = (w.mean() / w.std()) * sqrt_ppy
        axes[1].plot(dates, rolling_sh, color=color, linewidth=1.6,
                     marker=marker, markersize=4,
                     label=f"{name}  (mean = {np.nanmean(rolling_sh):.3f})")

        cum_sh = np.full(len(log_rets), np.nan)
        for j in range(len(log_rets)):
            w = log_rets[: j + 1]
            if len(w) >= 2 and w.std() > 1e-10:
                cum_sh[j] = (w.mean() / w.std()) * sqrt_ppy
        axes[2].plot(dates, cum_sh, color=color, linewidth=1.8,
                     marker=marker, markersize=4,
                     label=f"{name}  (final = {cum_sh[-1]:.3f})")

    for ax in axes:
        ax.axhline(0, color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.6)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Expected Sharpe\n(in-sample, per period)", fontsize=10)
    axes[0].legend(fontsize=9, loc="best", frameon=False)

    axes[1].set_ylabel(f"Rolling realised Sharpe\n"
                       f"(annualised, window = {rolling_window})", fontsize=10)
    axes[1].legend(fontsize=9, loc="best", frameon=False)

    axes[2].set_ylabel("Cumulative realised Sharpe\n(annualised, since start)",
                       fontsize=10)
    axes[2].legend(fontsize=9, loc="best", frameon=False)
    axes[2].set_xlabel(
        f"Rebalancing date  |  {cfg.rebalancing}",
        fontsize=10,
    )

    tick_interval = 1 if cfg.rebalancing == "quarterly" else 2
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=tick_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    _finalize(fig, save_path=save_path, show=show)


def plot_risk_return_evolution(results: dict, instance, cfg, period: int,
                               save_path: str | None = None, show: bool = True):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), dpi=130, sharex=True)

    best_name, best_period = None, None
    best_value = -np.inf
    for name, result in results.items():
        for p in result.periods:
            if p.portfolio_value > best_value:
                best_value = p.portfolio_value
                best_name, best_period = name, p

    if best_period is not None:
        sel = np.where(best_period.shares > 0)[0]
        composition = ", ".join(
            f"{instance.assets[i]}:{int(best_period.shares[i])}" for i in sel
        )
        fig.suptitle(
            f"Best Portfolio  ({best_name})  "
            f"{str(best_period.date)[:10]}  →  value = {best_value:,.0f}\n"
            f"{composition}",
            fontsize=11, fontweight="bold",
        )
    else:
        fig.suptitle("Risk / Return / Value over the back-test",
                     fontsize=12, fontweight="bold")

    for name, result in results.items():
        color  = COLORS.get(name, "gray")
        marker = MARKERS.get(name, "o")
        dates  = result.dates
        if len(dates) == 0:
            continue

        axes[0].plot(dates, result.avg_returns, color=color, linewidth=1.5,
                     marker=marker, markersize=4, alpha=0.9,
                     label=f"{name}  (avg yield = {result.avg_returns.mean():.4f})")

        axes[1].plot(dates, result.avg_risks, color=color, linewidth=1.5,
                     marker=marker, markersize=4, alpha=0.9,
                     label=f"{name}  (avg risk = {result.avg_risks.mean():.4f})")

        axes[2].plot(dates, result.portfolio_values, color=color, linewidth=1.8,
                     marker=marker, markersize=4,
                     label=f"{name}  (final = {result.portfolio_values[-1]:,.0f})")

    axes[0].set_ylabel("Expected return\n(avg over candidates)", fontsize=10)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=9, loc="best", frameon=False)

    axes[1].set_ylabel("Risk\n(avg over candidates)", fontsize=10)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=9, loc="best", frameon=False)

    axes[2].set_ylabel("Portfolio value", fontsize=10)
    axes[2].axhline(cfg.budget, color="gray", linewidth=0.8,
                    linestyle="--", alpha=0.6, label=f"Initial = {cfg.budget:,.0f}")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=9, loc="best", frameon=False)
    axes[2].set_xlabel(
        f"Date  |  rebalancing = {cfg.rebalancing} "
        f"(period = {period} {instance.frequency.lower()} rows)",
        fontsize=10,
    )

    tick_interval = 1 if cfg.rebalancing == "quarterly" else 2
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=tick_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    _finalize(fig, save_path=save_path, show=show)
