import argparse
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


DATA_DIR  = "data/stock/WEEK"
OUT_DIR   = "results/plots/dataset_description"
PRICE_COL = "Close"

EVENTS = [
    ("2020-02-23", "Inicio caída COVID"),
    ("2020-03-23", "Mínimo COVID"),
    ("2020-11-09", "Anuncio vacuna Pfizer"),
    ("2022-02-24", "Invasión de Ucrania"),
    ("2022-06-13", "Inflación pico USA"),
]


def load_universe(data_dir: str, price_col: str = "Close") -> pd.DataFrame:
    """Load all asset CSVs into a price matrix aligned by row position.

    In: data_dir (str), price_col (str).
    Out: pd.DataFrame (dates x assets).
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    series = []
    tickers = []
    dates_per_asset = []
    for fname in files:
        ticker = fname[:-4]
        df = pd.read_csv(os.path.join(data_dir, fname))
        prices = df[price_col].astype(float).reset_index(drop=True)
        dates = pd.to_datetime(df["Date"]).reset_index(drop=True)
        tickers.append(ticker)
        series.append(prices.rename(ticker))
        dates_per_asset.append(dates)

    n = min(len(s) for s in series)
    series = [s.iloc[:n] for s in series]
    prices = pd.concat(series, axis=1)
    canonical_dates = dates_per_asset[0].iloc[:n]
    prices.index = pd.DatetimeIndex(canonical_dates)
    return prices


def _save(fig, path: str, dpi: int = 140) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  saved {path}")


def plot_market_evolution(prices: pd.DataFrame, out_dir: str) -> None:
    norm = prices / prices.iloc[0]
    avg = norm.mean(axis=1)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=140)

    for col in norm.columns:
        ax.plot(norm.index, norm[col], color="lightgrey",
                linewidth=0.6, alpha=0.55)

    ax.plot(norm.index, avg, color="black", linewidth=2.3,
            label="Cartera equiponderada 1/N (media)")

    ymin, ymax = ax.get_ylim()
    for date_str, label in EVENTS:
        d = pd.Timestamp(date_str)
        if d < norm.index[0] or d > norm.index[-1]:
            continue
        ax.axvline(d, color="firebrick", linestyle="--",
                   alpha=0.55, linewidth=0.9)
        ax.text(d, ymax, " " + label, rotation=90, fontsize=8,
                va="top", ha="left", color="firebrick")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Evolución del mercado — precios normalizados (base = 1, primer cierre)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Valor normalizado")
    ax.set_xlabel("Fecha")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    _save(fig, os.path.join(out_dir, "01_market_evolution.png"))


def plot_returns_distribution(returns: pd.DataFrame, out_dir: str) -> None:
    all_rets = returns.values.flatten()
    all_rets = all_rets[np.isfinite(all_rets)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=140)

    axes[0].hist(all_rets, bins=70, color="steelblue",
                 edgecolor="black", alpha=0.85)
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axvline(float(np.mean(all_rets)), color="firebrick",
                    linewidth=1.4,
                    label=f"Media = {np.mean(all_rets):.4f}")
    axes[0].axvline(float(np.median(all_rets)), color="seagreen",
                    linewidth=1.4, linestyle="--",
                    label=f"Mediana = {np.median(all_rets):.4f}")
    axes[0].set_title("Distribución de retornos logarítmicos semanales",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Retorno logarítmico")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    rets_by_ticker = [returns[c].dropna().values for c in returns.columns]
    bp = axes[1].boxplot(rets_by_ticker, showfliers=True, patch_artist=True)
    axes[1].set_xticks(range(1, len(returns.columns) + 1))
    axes[1].set_xticklabels(list(returns.columns))
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
        patch.set_alpha(0.85)
        patch.set_linewidth(0.6)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(0.6)
    for cap in bp["caps"]:
        cap.set_linewidth(0.6)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("Dispersión por activo",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Activo")
    axes[1].set_ylabel("Retorno logarítmico semanal")
    plt.setp(axes[1].get_xticklabels(), rotation=90, fontsize=6)
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "02_returns_distribution.png"))


def plot_rolling_volatility(returns: pd.DataFrame, out_dir: str,
                            window: int = 8) -> None:
    eq = returns.mean(axis=1)
    rolling_vol = eq.rolling(window=window).std() * np.sqrt(52)
    cross_section_vol = returns.std(axis=1) * np.sqrt(52)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
    ax.plot(rolling_vol.index, rolling_vol.values, color="darkorange",
            linewidth=1.9,
            label=f"Volatilidad anualizada del 1/N (ventana {window} sem.)")
    ax.fill_between(rolling_vol.index, 0, rolling_vol.values,
                    alpha=0.15, color="darkorange")
    ax.plot(cross_section_vol.index, cross_section_vol.values,
            color="seagreen", linewidth=1.0, alpha=0.7,
            label="Dispersión transversal anualizada (semanal)")
    for date_str, label in EVENTS:
        d = pd.Timestamp(date_str)
        if d < rolling_vol.index[0] or d > rolling_vol.index[-1]:
            continue
        ax.axvline(d, color="firebrick", linestyle="--",
                   alpha=0.45, linewidth=0.8)
    ax.set_title(f"Evolución de la volatilidad — ventana móvil {window} semanas",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Volatilidad anualizada")
    ax.set_xlabel("Fecha")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    _save(fig, os.path.join(out_dir, "03_rolling_volatility.png"))


def plot_drawdown(prices: pd.DataFrame, out_dir: str) -> None:
    norm = prices / prices.iloc[0]
    eq = norm.mean(axis=1)
    running_max = eq.cummax()
    drawdown = (eq / running_max) - 1.0

    fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color="firebrick", alpha=0.55)
    ax.plot(drawdown.index, drawdown.values, color="darkred", linewidth=1.4)

    min_idx = drawdown.idxmin()
    min_val = float(drawdown.min())
    ax.scatter([min_idx], [min_val], s=120, color="black", zorder=5)
    ax.annotate(f"Drawdown máximo: {min_val:.1%}\n{min_idx.date()}",
                xy=(min_idx, min_val), xytext=(15, 30),
                textcoords="offset points", fontsize=10,
                arrowprops=dict(arrowstyle="->", linewidth=0.9),
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white", edgecolor="gray"))
    for date_str, _ in EVENTS:
        d = pd.Timestamp(date_str)
        if d < drawdown.index[0] or d > drawdown.index[-1]:
            continue
        ax.axvline(d, color="black", linestyle="--",
                   alpha=0.25, linewidth=0.7)
    ax.set_title("Drawdown del índice 1/N (buy-and-hold equiponderado)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Fecha")
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    _save(fig, os.path.join(out_dir, "04_drawdown.png"))


def plot_correlation_heatmap(returns: pd.DataFrame, out_dir: str) -> None:
    corr = returns.corr()
    order = corr.mean().sort_values(ascending=False).index
    corr_ord = corr.loc[order, order]

    fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
    cmap = LinearSegmentedColormap.from_list(
        "rwb", ["#2166ac", "#f7f7f7", "#b2182b"])
    im = ax.imshow(corr_ord.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticklabels(order, fontsize=7)
    ax.set_title("Matriz de correlación de retornos semanales",
                 fontsize=13, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlación de Pearson")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "05_correlation_heatmap.png"))


def plot_best_worst_performers(prices: pd.DataFrame, out_dir: str,
                                top_k: int = 10) -> None:
    total_ret = (prices.iloc[-1] / prices.iloc[0] - 1).sort_values()
    bottom = total_ret.head(top_k)
    top = total_ret.tail(top_k)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=140)
    axes[0].barh(bottom.index, bottom.values * 100,
                 color="firebrick", edgecolor="black", alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=0.6)
    axes[0].set_title(f"Peores {top_k} activos — retorno total acumulado",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Retorno (%)")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(top.index, top.values * 100,
                 color="seagreen", edgecolor="black", alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].set_title(f"Mejores {top_k} activos — retorno total acumulado",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Retorno (%)")
    axes[1].grid(axis="x", alpha=0.25)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "06_best_worst_performers.png"))


def plot_annual_returns_heatmap(returns: pd.DataFrame, out_dir: str) -> None:
    annual = returns.groupby(returns.index.year).sum()

    fig, ax = plt.subplots(
        figsize=(11, max(4.5, 0.22 * len(annual.columns))), dpi=140,
    )
    vmax = float(np.nanmax(np.abs(annual.values)))
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#b2182b", "#f7f7f7", "#1b7837"])
    im = ax.imshow(annual.T.values, cmap=cmap,
                   vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(annual.index)))
    ax.set_xticklabels(annual.index)
    ax.set_yticks(range(len(annual.columns)))
    ax.set_yticklabels(annual.columns, fontsize=7)
    ax.set_title("Retorno log acumulado anual por activo",
                 fontsize=13, fontweight="bold", pad=12)
    for i, _ in enumerate(annual.index):
        for j, _ in enumerate(annual.columns):
            v = annual.iloc[i, j]
            ax.text(i, j, f"{v:.0%}", ha="center", va="center",
                    fontsize=6, color="black" if abs(v) < 0.4 else "white")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Retorno log anual")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "07_annual_returns.png"))


def plot_volume_evolution(data_dir: str, prices_index: pd.DatetimeIndex,
                          out_dir: str) -> None:
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    volumes = []
    for fname in files:
        df = pd.read_csv(os.path.join(data_dir, fname))
        vol = df["Volume"].astype(float).reset_index(drop=True)
        volumes.append(vol)
    n = min(len(v) for v in volumes)
    volumes = [v.iloc[:n].values for v in volumes]
    volume_total = np.sum(volumes, axis=0)[:len(prices_index)]
    dates = prices_index[:len(volume_total)]

    fig, ax = plt.subplots(figsize=(14, 5), dpi=140)
    ax.bar(dates, volume_total / 1e9, width=4.0,
           color="navy", alpha=0.75, edgecolor="none")
    rolling = pd.Series(volume_total / 1e9).rolling(window=4).mean()
    ax.plot(dates, rolling.values, color="darkorange",
            linewidth=2.0, label="Media móvil 4 semanas")
    for date_str, _ in EVENTS:
        d = pd.Timestamp(date_str)
        if d < dates[0] or d > dates[-1]:
            continue
        ax.axvline(d, color="firebrick", linestyle="--",
                   alpha=0.55, linewidth=0.9)
    ax.set_title("Volumen total agregado (billones de unidades) — todos los activos",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Volumen agregado (×10⁹)")
    ax.set_xlabel("Fecha")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    _save(fig, os.path.join(out_dir, "08_volume_evolution.png"))


def print_dataset_summary(prices: pd.DataFrame,
                          returns: pd.DataFrame) -> None:
    print("\n" + "═" * 67)
    print("  RESUMEN DEL DATASET")
    print("═" * 67)
    print(f"  Activos:              {prices.shape[1]}")
    print(f"  Observaciones:        {prices.shape[0]} semanas")
    print(f"  Fecha inicial:        {prices.index[0].date()}")
    print(f"  Fecha final:          {prices.index[-1].date()}")
    print(f"  Duración:             "
          f"{(prices.index[-1] - prices.index[0]).days / 365.25:.2f} años")
    print("  Frecuencia:           Semanal (close-to-close)")
    print()
    mean_ret = float(returns.mean().mean()) * 52
    vol = float(returns.std().mean()) * np.sqrt(52)
    eq = returns.mean(axis=1)
    sharpe = float(eq.mean() / eq.std()) * np.sqrt(52)
    total_ret = (prices.iloc[-1] / prices.iloc[0] - 1).sort_values()
    print(f"  Retorno anualizado medio:                 {mean_ret:.2%}")
    print(f"  Volatilidad anualizada media:             {vol:.2%}")
    print(f"  Sharpe del 1/N (anualizado, rf = 0):      {sharpe:.4f}")
    print(f"  Mejor activo:         {total_ret.index[-1]}  "
          f"({total_ret.iloc[-1]:.2%})")
    print(f"  Peor activo:          {total_ret.index[0]}  "
          f"({total_ret.iloc[0]:.2%})")
    print("═" * 67 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir",  default=OUT_DIR)
    parser.add_argument("--price-col", default=PRICE_COL)
    args = parser.parse_args()

    print(f"Loading universe from {args.data_dir} ...")
    prices = load_universe(args.data_dir, price_col=args.price_col)
    returns = np.log(prices / prices.shift(1)).iloc[1:]
    print(f"Loaded {prices.shape[1]} assets × {prices.shape[0]} weeks.")
    print(f"Output folder: {args.out_dir}\n")
    os.makedirs(args.out_dir, exist_ok=True)

    print_dataset_summary(prices, returns)

    print("Generating plots ...")
    plot_market_evolution(prices, args.out_dir)
    plot_returns_distribution(returns, args.out_dir)
    plot_rolling_volatility(returns, args.out_dir)
    plot_drawdown(prices, args.out_dir)
    plot_correlation_heatmap(returns, args.out_dir)
    plot_best_worst_performers(prices, args.out_dir)
    plot_annual_returns_heatmap(returns, args.out_dir)
    plot_volume_evolution(args.data_dir, prices.index, args.out_dir)
    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
