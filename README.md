# Crypto Market Making & Microstructure

Personal project reconstructing a live limit order book from raw exchange
data and using it to estimate microstructure measures and backtest a
simple market maker, on BTC/USDT.

## Data

Live-captured via the Binance.US WebSocket diff-depth stream
(`depth@100ms`) plus a REST snapshot, one continuous 13-hour session,
single segment (no reconnects, no sequence gaps). Raw JSONL (snapshot,
depth events, trade events) kept alongside the reconstructed parquet
outputs.

## Structure

```
src/
  acquisition.py     WebSocket + REST capture, snapshot/diff reconciliation
  reconstruct.py     Numba-accelerated book reconstruction from raw events
  measures/          kyle_lambda.py, vpin.py
  market_making/     backtest.py
data/
  raw/               snapshots.jsonl, depth_events.jsonl, trade_events.jsonl
  clean/             book_summary.parquet, trade_summary.parquet
output/              regression / backtest results, CSVs
```

## Pipeline

**Acquisition** - opens a WebSocket depth stream, fetches a REST snapshot,
and validates diffs against Binance's official reconciliation algorithm
(`U`/`u` sequence chaining, stale-event discarding). Segmented so a broken
chain restarts cleanly instead of silently drifting.

**Reconstruction** - two-stage: raw JSON parsed and validated in Python,
then a Numba-jitted loop replays the event stream against a live bid/ask
book (absolute-quantity updates, zero = remove level), emitting a compact
per-event summary (best bid/ask, spread, midprice, top-10 depth,
imbalance) instead of full book snapshots.

**Measures** - Kyle's lambda (price impact of signed order flow, 1-min
bins, sqrt-dampened volume) and VPIN (volume-synchronized order-flow
toxicity, equal-volume buckets, rolling window).

**Analysis** - tests whether order-book imbalance predicts short-horizon
midprice moves (1s / 5s / 30s), HAC-adjusted for overlapping windows.

**Market making** - simple maker quoting mid +/- half-spread, spread
widened when VPIN is elevated, hard inventory cap, fills approximated by
trade-tape crossing. P&L split into spread-capture vs.
inventory/adverse-selection components.

## Results

Kyle's lambda (13-hr session, 1-min bins, HAC/robust SE):

- lambda = 13.72, p = 0.004 - significant, correctly-signed price impact
- Time-varying: significant in ~7 of 13 hourly sub-windows, magnitude
  swinging roughly 4-135 across hours

VPIN (bucket-size sensitivity check):

- 200 buckets: mean 0.81; 50 buckets: mean 0.71 - some small-bucket bias,
  but imbalance persists at coarser granularity, consistent with thin
  Binance.US liquidity relative to major venues

Order-book imbalance -> future midprice (46.8k obs, HAC SE):

- Null at all three horizons (p = 0.39 / 0.48 / 0.91) - no significant
  short-horizon predictive power from top-10 depth imbalance

Market-making backtest (data-driven half-spread, VPIN-aware widening,
0.05 BTC inventory cap):

- Total P&L $45.96, but only $3.04 from spread capture - the rest is
  unhedged inventory drift, not repeatable edge
- Sharpe (raw, per-min equity) = 0.03 - statistically indistinguishable
  from zero; no demonstrated edge on this single session