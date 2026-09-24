"""
Design choices:
- delta_P = change in midprice per 1-min bin (not trade price - avoids bid-ask bounce, defined even in zero-trade bins).
- S = sum of sign(qty) * sqrt(|qty|) per bin (dampens outlier trade sizes).
- Zero-trade bins kept (S=0), not dropped.
- Primary: single OLS over the full session. Secondary: lambda re-estimated per hour as a diagnostic for drift over the session.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

CLEAN_DIR = Path("../data/clean")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_FREQ = "1min"


def load_data():
    book_df = pd.read_parquet(CLEAN_DIR / "book_summary.parquet")
    trade_df = pd.read_parquet(CLEAN_DIR / "trade_summary.parquet")
    book_df["ts"] = pd.to_datetime(book_df["local_time"], unit="s")
    trade_df["ts"] = pd.to_datetime(trade_df["local_time"], unit="s")
    return book_df, trade_df

def build_bins(book_df: pd.DataFrame, trade_df: pd.DataFrame) -> pd.DataFrame:
    midprice = (
        book_df.set_index("ts")["midprice"]
        .resample(BIN_FREQ)
        .last()
        .ffill()
    )
    delta_p = midprice.diff()

    trade_df = trade_df.copy()
    trade_df["s_component"] = np.sign(trade_df["signed_qty"]) * np.sqrt(
        trade_df["qty"]
    )
    S = trade_df.set_index("ts")["s_component"].resample(BIN_FREQ).sum()

    df = pd.DataFrame({"delta_p": delta_p, "S": S}).reindex(midprice.index)
    df["S"] = df["S"].fillna(0.0)
    df = df.dropna(subset=["delta_p"])
    return df

def run_ols(df: pd.DataFrame):
    X = sm.add_constant(df["S"])
    y = df["delta_p"]
    model = sm.OLS(y, X, missing="drop").fit()
    return model

def hourly_lambda(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hour, group in df.groupby(df.index.floor("h")):
        if len(group) < 10:
            continue
        model = run_ols(group)
        rows.append(
            {
                "hour": hour,
                "n_bins": len(group),
                "lambda": model.params["S"],
                "se": model.bse["S"],
                "t_stat": model.tvalues["S"],
                "p_value": model.pvalues["S"],
                "r_squared": model.rsquared,
            }
        )
    return pd.DataFrame(rows)

def main():
    book_df, trade_df = load_data()
    df = build_bins(book_df, trade_df)

    overall_model = run_ols(df)
    print(overall_model.summary())

    overall_result = pd.DataFrame(
        [
            {
                "n_bins": len(df),
                "lambda": overall_model.params["S"],
                "se": overall_model.bse["S"],
                "t_stat": overall_model.tvalues["S"],
                "p_value": overall_model.pvalues["S"],
                "r_squared": overall_model.rsquared,
            }
        ]
    )
    overall_result.to_csv(OUTPUT_DIR / "kyle_lambda_overall.csv", index=False)

    hourly_df = hourly_lambda(df)
    hourly_df.to_csv(OUTPUT_DIR / "kyle_lambda_hourly.csv", index=False)

    print(f"\noverall lambda: {overall_model.params['S']:.6f}")
    print(f"\nhourly diagnostic:\n{hourly_df}")


if __name__ == "__main__":
    main()