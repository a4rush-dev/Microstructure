"""
VPIN: volume-synchronized probability of informed trading.

Design choices:
- Trades split into B equal-volume buckets (not time-based) - a single
  trade can straddle a bucket boundary and gets split across both.
- Each bucket's buy/sell volume comes from signed_qty (already
  classified via isBuyerMaker in reconstruct.py).
- VPIN = rolling mean over the last n buckets of |V_buy - V_sell| / V,
  giving B - n + 1 VPIN readings across the session.
- B=200, n=50 are defaults (~7 trades/bucket, 151 readings on ~13h of
  data) - adjust if bucket contents look too thin/thick once run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path("../data/clean")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BUCKETS = 50
ROLLING_WINDOW = 25

def load_trades() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_DIR / "trade_summary.parquet")
    df = df.sort_values("local_time").reset_index(drop=True)
    df["buy_qty"] = np.where(df["signed_qty"] > 0, df["qty"], 0.0)
    df["sell_qty"] = np.where(df["signed_qty"] < 0, df["qty"], 0.0)
    return df

def build_volume_buckets(df: pd.DataFrame, n_buckets: int) -> pd.DataFrame:
    total_volume = df["qty"].sum()
    bucket_size = total_volume / n_buckets
    print(f"total volume: {total_volume:.4f}, bucket size: {bucket_size:.6f}")

    buckets = []
    cum_in_bucket = 0.0
    buy_acc = 0.0
    sell_acc = 0.0
    bucket_end_time = None

    for row in df.itertuples():
        remaining_qty = row.qty
        buy_frac = row.buy_qty / row.qty if row.qty > 0 else 0.0
        sell_frac = row.sell_qty / row.qty if row.qty > 0 else 0.0

        while remaining_qty > 0:
            space_left = bucket_size - cum_in_bucket
            take = min(space_left, remaining_qty)

            buy_acc += take * buy_frac
            sell_acc += take * sell_frac
            cum_in_bucket += take
            remaining_qty -= take
            bucket_end_time = row.local_time

            if cum_in_bucket >= bucket_size - 1e-12:
                buckets.append(
                    {
                        "bucket_end_time": bucket_end_time,
                        "buy_volume": buy_acc,
                        "sell_volume": sell_acc,
                    }
                )
                cum_in_bucket = 0.0
                buy_acc = 0.0
                sell_acc = 0.0

                if len(buckets) >= n_buckets:
                    break
        if len(buckets) >= n_buckets:
            break

    bucket_df = pd.DataFrame(buckets)
    bucket_df["bucket_size"] = bucket_size
    return bucket_df


def compute_vpin(bucket_df: pd.DataFrame, window: int) -> pd.DataFrame:
    imbalance = (bucket_df["buy_volume"] - bucket_df["sell_volume"]).abs()
    vpin = imbalance.rolling(window).sum() / (window * bucket_df["bucket_size"])
    bucket_df = bucket_df.copy()
    bucket_df["vpin"] = vpin
    return bucket_df.dropna(subset=["vpin"]).reset_index(drop=True)


def main():
    trades = load_trades()
    bucket_df = build_volume_buckets(trades, N_BUCKETS)
    print(f"actual buckets built: {len(bucket_df)} (requested {N_BUCKETS})")

    vpin_df = compute_vpin(bucket_df, ROLLING_WINDOW)
    vpin_df.to_csv(OUTPUT_DIR / "vpin_series.csv", index=False)

    print(f"\nVPIN readings: {len(vpin_df)}")
    print(vpin_df["vpin"].describe())


if __name__ == "__main__":
    main()