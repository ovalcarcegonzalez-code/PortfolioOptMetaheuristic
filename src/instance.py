from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Instance:
    prices: pd.DataFrame
    returns: pd.DataFrame
    assets: list[str]
    frequency: str
    periods_per_year: int
    price_col: str

    @classmethod
    def from_folder(
        cls,
        data_root: str,
        frequency: str = "WEEK",
        price_col: str = "Close",
    ):
        """Build an Instance from one CSV per asset, aligned by row position.

        In: data_root (str), frequency (str), price_col (str).
        Out: Instance (assets, dates, price matrix).
        """
        frequency = frequency.upper()
        if frequency not in {"DAY", "WEEK"}:
            raise ValueError("frequency must be 'DAY' or 'WEEK'.")

        periods_per_year = {"DAY": 252, "WEEK": 52}[frequency]

        folder = Path(data_root) / frequency
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {folder}")

        assets: list[str] = []
        per_asset_prices: list[pd.Series] = []
        per_asset_dates:  list[pd.Series] = []

        for path in csv_files:
            asset = path.stem
            df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
            if price_col not in df.columns:
                raise ValueError(
                    f"Column '{price_col}' not found in {path.name}. "
                    f"Available: {list(df.columns)}"
                )
            assets.append(asset)
            per_asset_prices.append(df[price_col].reset_index(drop=True).rename(asset))
            per_asset_dates.append(df["Date"].reset_index(drop=True))

        n = min(len(s) for s in per_asset_prices)
        per_asset_prices = [s.iloc[:n] for s in per_asset_prices]
        canonical_dates = per_asset_dates[0].iloc[:n]

        prices = pd.concat(per_asset_prices, axis=1)
        prices.index = pd.DatetimeIndex(canonical_dates)
        prices.index.name = "Date"

        if prices.isna().any().any():
            prices = prices.ffill().bfill()

        returns = np.log(prices / prices.shift(1)).dropna()

        return cls(
            prices=prices,
            returns=returns,
            assets=assets,
            frequency=frequency,
            periods_per_year=periods_per_year,
            price_col=price_col,
        )
