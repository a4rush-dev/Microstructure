"""
Simple market-making backtest on reconstructed book + trade data.

Design choices:
- Fill assumption: a quote is filled whenever a trade crosses it
  (price <= quoted_bid for a sell-initiated trade, price >= quoted_ask
  for a buy-initiated trade). No queue priority modeled - optimistic.
- Instant re-quoting: quotes = midprice +/- half_spread at the moment
  of each trade, no latency.
- half_spread = mean observed market spread / 2 (data-driven).
- VPIN enhancement: when the most recent VPIN reading is above the
  75th percentile of the VPIN series, half_spread is doubled.
- Inventory cap: 0.05 BTC. Quoting on a side stops once inventory
  would breach the cap on that side.
- P&L decomposition: spread_pnl = half_spread * qty at each fill
  (the "if price never moved" component). adverse_selection_pnl =
  total_pnl - spread_pnl (the rest - price moving against inventory).
- Sharpe computed on a 1-minute mark-to-market equity curve, raw
  (unannualized) - the sample is 13 hours, annualizing would overstate
  precision.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path("../data/clean")
OUTPUT_DIR = Path("../measures/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VPIN_PERCENTILE_THRESHOLD = 0.75
VPIN_WIDEN_FACTOR = 2.0
INVENTORY_CAP_BTC = 0.05


def load_data():
    book_df = pd.read_parquet(CLEAN_DIR / "book_summary.parquet")
    trade_df = pd.read_parquet(CLEAN_DIR / "trade_summary.parquet")
    vpin_df = pd.read_csv(OUTPUT_DIR / "vpin_series.csv")

    book_df["ts"] = pd.to_datetime(book_df["local_time"], unit="s")
    trade_df["ts"] = pd.to_datetime(trade_df["local_time"], unit="s")
    vpin_df["ts"] = pd.to_datetime(vpin_df["bucket_end_time"], unit="s")

    return (
        book_df.sort_values("ts").reset_index(drop=True),
        trade_df.sort_values("ts").reset_index(drop=True),
        vpin_df.sort_values("ts").reset_index(drop=True),
    )


def attach_reference_state(trade_df, book_df, vpin_df):
    df = pd.merge_asof(
        trade_df, book_df[["ts", "midprice"]], on="ts", direction="backward"
    )
    df = pd.merge_asof(
        df, vpin_df[["ts", "vpin"]], on="ts", direction="backward"
    )
    return df.dropna(subset=["midprice"])


def run_backtest(trade_df: pd.DataFrame, base_half_spread: float, vpin_threshold: float):
    inventory = 0.0
    cash = 0.0
    spread_pnl = 0.0
    fills = []
    equity_points = []

    for row in trade_df.itertuples():
        widen = row.vpin is not None and not pd.isna(row.vpin) and row.vpin > vpin_threshold
        half_spread = base_half_spread * VPIN_WIDEN_FACTOR if widen else base_half_spread

        quoted_bid = row.midprice - half_spread
        quoted_ask = row.midprice + half_spread

        is_sell_initiated = row.signed_qty < 0

        if is_sell_initiated and row.price <= quoted_bid and inventory < INVENTORY_CAP_BTC:
            fill_qty = min(row.qty, INVENTORY_CAP_BTC - inventory)
            cash -= fill_qty * quoted_bid
            inventory += fill_qty
            spread_pnl += fill_qty * half_spread
            fills.append({"ts": row.ts, "side": "buy", "qty": fill_qty, "price": quoted_bid})

        elif (not is_sell_initiated) and row.price >= quoted_ask and inventory > -INVENTORY_CAP_BTC:
            fill_qty = min(row.qty, INVENTORY_CAP_BTC + inventory)
            cash += fill_qty * quoted_ask
            inventory -= fill_qty
            spread_pnl += fill_qty * half_spread
            fills.append({"ts": row.ts, "side": "sell", "qty": fill_qty, "price": quoted_ask})

        equity_points.append(
            {"ts": row.ts, "cash": cash, "inventory": inventory, "midprice": row.midprice}
        )

    fills_df = pd.DataFrame(fills)
    equity_df = pd.DataFrame(equity_points)
    equity_df["equity"] = equity_df["cash"] + equity_df["inventory"] * equity_df["midprice"]

    return fills_df, equity_df, spread_pnl


def compute_sharpe(equity_df: pd.DataFrame) -> float:
    minute_equity = equity_df.set_index("ts")["equity"].resample("1min").last().ffill()
    minute_returns = minute_equity.diff().dropna()
    if minute_returns.std() == 0:
        return np.nan
    return minute_returns.mean() / minute_returns.std()


def main():
    book_df, trade_df, vpin_df = load_data()
    trade_df = attach_reference_state(trade_df, book_df, vpin_df)

    base_half_spread = book_df["spread"].mean() / 2
    vpin_threshold = vpin_df["vpin"].quantile(VPIN_PERCENTILE_THRESHOLD)

    print(f"base half-spread: {base_half_spread:.4f}")
    print(f"VPIN widen threshold ({VPIN_PERCENTILE_THRESHOLD:.0%}ile): {vpin_threshold:.4f}")

    fills_df, equity_df, spread_pnl = run_backtest(trade_df, base_half_spread, vpin_threshold)

    total_pnl = equity_df["equity"].iloc[-1]
    adverse_selection_pnl = total_pnl - spread_pnl
    sharpe = compute_sharpe(equity_df)

    print(f"\nfills: {len(fills_df)}")
    print(f"total P&L: ${total_pnl:.4f}")
    print(f"spread-capture P&L: ${spread_pnl:.4f}")
    print(f"adverse-selection P&L: ${adverse_selection_pnl:.4f}")
    print(f"Sharpe (raw, per-minute equity): {sharpe:.4f}")

    fills_df.to_csv(Path("output") / "mm_fills.csv", index=False)
    equity_df.to_csv(Path("output") / "mm_equity_curve.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "n_fills": len(fills_df),
                "total_pnl": total_pnl,
                "spread_pnl": spread_pnl,
                "adverse_selection_pnl": adverse_selection_pnl,
                "sharpe_raw": sharpe,
                "base_half_spread": base_half_spread,
                "vpin_threshold": vpin_threshold,
                "inventory_cap": INVENTORY_CAP_BTC,
            }
        ]
    )
    summary.to_csv(Path("output") / "mm_backtest_summary.csv", index=False)


if __name__ == "__main__":
    main()