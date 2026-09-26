"""
Order-book imbalance -> future midprice change, at multiple horizons.

Design choices:
- book_summary resampled to a 1-second grid (last value per second,
  forward-filled) - imbalance and midprice both need a common clock.
- Horizons tested: 1s, 5s, 30s. future_return_h = midprice(t+h) - midprice(t).
- Regression: future_return_h ~ imbalance_t, one regression per horizon.
- HAC (Newey-West) standard errors, maxlags=h - overlapping horizon
  windows make consecutive observations autocorrelated, which a plain
  OLS or even HC1 correction does not fix. maxlags=h is the standard
  choice since that's the width of the overlap.
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

CLEAN_DIR = Path("../data/clean")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_SEC = [1, 5, 30]


def load_grid() -> pd.DataFrame:
    book_df = pd.read_parquet(CLEAN_DIR / "book_summary.parquet")
    book_df["ts"] = pd.to_datetime(book_df["local_time"], unit="s")
    grid = (
        book_df.set_index("ts")[["midprice", "imbalance"]]
        .resample("1s")
        .last()
        .ffill()
    )
    return grid

def run_horizon_regression(grid: pd.DataFrame, horizon: int):
    df = grid.copy()
    df["future_return"] = df["midprice"].shift(-horizon) - df["midprice"]
    df = df.dropna(subset=["future_return", "imbalance"])

    X = sm.add_constant(df["imbalance"])
    y = df["future_return"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
    return model, len(df)


def main():
    grid = load_grid()
    print(f"grid rows (1s bins): {len(grid)}")

    results = []
    for h in HORIZONS_SEC:
        model, n = run_horizon_regression(grid, h)
        print(f"\n--- horizon = {h}s ---")
        print(model.summary())
        results.append(
            {
                "horizon_sec": h,
                "n_obs": n,
                "coef": model.params["imbalance"],
                "se": model.bse["imbalance"],
                "t_stat": model.tvalues["imbalance"],
                "p_value": model.pvalues["imbalance"],
                "r_squared": model.rsquared,
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "imbalance_predictive_regressions.csv", index=False)
    print(f"\nsummary across horizons:\n{results_df}")


if __name__ == "__main__":
    main()