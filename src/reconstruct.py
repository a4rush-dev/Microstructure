"""
Design choices:

- parse JSONL, reconcile Binance diffs per segment, validate continuity, and flatten to NumPy arrays.
- Numba hot loop maintains the book in typed dicts, applies absolute-quantity updates, and emits one summary row per depth event.
- Price levels use integer scaling to avoid float-equality issues; zero quantity deletes a level.
- Segments are truncated at continuity gaps rather than silently continuing with a corrupted book.
- Depth summary includes timestamps, best bid/ask, spread, midprice, top-DEPTH_LEVELS depth, and order-book imbalance.
- Trade summary is a vectorized transform of signed quantity, price, and timestamps.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import int64, float64, njit
from numba.typed import Dict

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_SCALE = 100_000_000
DEPTH_LEVELS = 10


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_price_int(price_str: str) -> int:
    return int(round(float(price_str) * PRICE_SCALE))

def build_initial_book(snapshot: dict):
    bids = Dict.empty(key_type=int64, value_type=float64)
    asks = Dict.empty(key_type=int64, value_type=float64)

    for price_str, qty_str in snapshot["bids"]:
        qty = float(qty_str)
        if qty > 0:
            bids[to_price_int(price_str)] = qty

    for price_str, qty_str in snapshot["asks"]:
        qty = float(qty_str)
        if qty > 0:
            asks[to_price_int(price_str)] = qty

    return bids, asks

def align_segment_events(snapshot_record: dict, depth_records: list) -> list:
    last_update_id = snapshot_record["snapshot"]["lastUpdateId"]

    kept = [r for r in depth_records if r["data"]["u"] > last_update_id]
    if not kept:
        return []

    first = kept[0]["data"]
    if not (first["U"] <= last_update_id + 1 <= first["u"]):
        raise ValueError(
            f"segment {snapshot_record['segment_id']}: snapshot does not "
            f"overlap cleanly with first kept event (lastUpdateId="
            f"{last_update_id}, first U={first['U']}, first u={first['u']})"
        )

    aligned = [kept[0]]
    for prev, curr in zip(kept, kept[1:]):
        if curr["data"]["U"] != prev["data"]["u"] + 1:
            print(
                f"segment {snapshot_record['segment_id']}: gap detected "
                f"(expected U={prev['data']['u'] + 1}, got U={curr['data']['U']}), "
                f"truncating segment here"
            )
            break
        aligned.append(curr)

    return aligned

def flatten_events(aligned_events: list):
    event_idx_list = []
    side_list = []
    price_list = []
    qty_list = []
    event_local_time = []
    event_exchange_time = []

    for i, record in enumerate(aligned_events):
        data = record["data"]
        event_local_time.append(record["local_time"])
        event_exchange_time.append(data.get("E", 0))

        for price_str, qty_str in data["b"]:
            event_idx_list.append(i)
            side_list.append(0)
            price_list.append(to_price_int(price_str))
            qty_list.append(float(qty_str))

        for price_str, qty_str in data["a"]:
            event_idx_list.append(i)
            side_list.append(1)
            price_list.append(to_price_int(price_str))
            qty_list.append(float(qty_str))

    return (
        np.array(event_idx_list, dtype=np.int64),
        np.array(side_list, dtype=np.int8),
        np.array(price_list, dtype=np.int64),
        np.array(qty_list, dtype=np.float64),
        np.array(event_local_time, dtype=np.float64),
        np.array(event_exchange_time, dtype=np.int64),
        len(aligned_events),
    )


@njit(cache=True)
def reconstruct_book(
    bids, asks, event_idx, side, price, qty, n_events, depth_levels
):
    out_best_bid_px = np.zeros(n_events, dtype=np.float64)
    out_best_bid_qty = np.zeros(n_events, dtype=np.float64)
    out_best_ask_px = np.zeros(n_events, dtype=np.float64)
    out_best_ask_qty = np.zeros(n_events, dtype=np.float64)
    out_spread = np.zeros(n_events, dtype=np.float64)
    out_mid = np.zeros(n_events, dtype=np.float64)
    out_bid_depth = np.zeros(n_events, dtype=np.float64)
    out_ask_depth = np.zeros(n_events, dtype=np.float64)
    out_imbalance = np.zeros(n_events, dtype=np.float64)

    n_rows = event_idx.shape[0]
    row = 0

    for e in range(n_events):
        while row < n_rows and event_idx[row] == e:
            p = price[row]
            q = qty[row]
            if side[row] == 0:
                if q == 0.0:
                    if p in bids:
                        del bids[p]
                else:
                    bids[p] = q
            else:
                if q == 0.0:
                    if p in asks:
                        del asks[p]
                else:
                    asks[p] = q
            row += 1

        n_bids = len(bids)
        n_asks = len(asks)

        if n_bids == 0 or n_asks == 0:
            print("am i here")
            continue

        bid_keys = np.empty(n_bids, dtype=np.int64)
        i = 0
        for k in bids:
            bid_keys[i] = k
            i += 1
        bid_keys.sort()

        ask_keys = np.empty(n_asks, dtype=np.int64)
        i = 0
        for k in asks:
            ask_keys[i] = k
            i += 1
        ask_keys.sort()

        best_bid_key = bid_keys[-1]
        best_ask_key = ask_keys[0]

        best_bid_px = best_bid_key / PRICE_SCALE
        best_ask_px = best_ask_key / PRICE_SCALE

        n_bid_top = min(depth_levels, n_bids)
        n_ask_top = min(depth_levels, n_asks)

        bid_depth = 0.0
        for j in range(n_bid_top):
            bid_depth += bids[bid_keys[n_bids - 1 - j]]

        ask_depth = 0.0
        for j in range(n_ask_top):
            ask_depth += asks[ask_keys[j]]

        out_best_bid_px[e] = best_bid_px
        out_best_bid_qty[e] = bids[best_bid_key]
        out_best_ask_px[e] = best_ask_px
        out_best_ask_qty[e] = asks[best_ask_key]
        out_spread[e] = best_ask_px - best_bid_px
        out_mid[e] = (best_ask_px + best_bid_px) / 2.0
        out_bid_depth[e] = bid_depth
        out_ask_depth[e] = ask_depth
        denom = bid_depth + ask_depth
        out_imbalance[e] = (bid_depth - ask_depth) / denom if denom > 0 else 0.0

    return (
        out_best_bid_px,
        out_best_bid_qty,
        out_best_ask_px,
        out_best_ask_qty,
        out_spread,
        out_mid,
        out_bid_depth,
        out_ask_depth,
        out_imbalance,
    )


def process_segment(snapshot_record: dict, depth_records: list) -> pd.DataFrame:
    aligned = align_segment_events(snapshot_record, depth_records)
    if not aligned:
        return pd.DataFrame()

    bids, asks = build_initial_book(snapshot_record["snapshot"])
    (
        event_idx,
        side,
        price,
        qty,
        event_local_time,
        event_exchange_time,
        n_events,
    ) = flatten_events(aligned)

    results = reconstruct_book(
        bids, asks, event_idx, side, price, qty, n_events, DEPTH_LEVELS
    )

    df = pd.DataFrame(
        {
            "local_time": event_local_time,
            "exchange_time": event_exchange_time,
            "best_bid_px": results[0],
            "best_bid_qty": results[1],
            "best_ask_px": results[2],
            "best_ask_qty": results[3],
            "spread": results[4],
            "midprice": results[5],
            "bid_depth": results[6],
            "ask_depth": results[7],
            "imbalance": results[8],
        }
    )
    return df


def build_trade_summary(trade_records: list) -> pd.DataFrame:
    rows = []
    for record in trade_records:
        data = record["data"]
        qty = float(data["q"])
        is_buyer_maker = data["m"]
        signed_qty = -qty if is_buyer_maker else qty
        rows.append(
            {
                "local_time": record["local_time"],
                "exchange_time": data.get("T", 0),
                "price": float(data["p"]),
                "qty": qty,
                "signed_qty": signed_qty,
                "is_buyer_maker": is_buyer_maker,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    snapshots = load_jsonl(RAW_DIR / "snapshots.jsonl")
    depth_events = load_jsonl(RAW_DIR / "depth_events.jsonl")
    trade_events = load_jsonl(RAW_DIR / "trade_events.jsonl")

    segment_ids = sorted({s["segment_id"] for s in snapshots})
    all_book_frames = []

    for seg_id in segment_ids:
        snapshot_record = next(s for s in snapshots if s["segment_id"] == seg_id)
        seg_depth_records = [d for d in depth_events if d["segment_id"] == seg_id]

        print(f"segment {seg_id}: {len(seg_depth_records)} raw depth events")
        df = process_segment(snapshot_record, seg_depth_records)
        print(f"segment {seg_id}: {len(df)} reconstructed book states")

        df["segment_id"] = seg_id
        all_book_frames.append(df)

    book_df = pd.concat(all_book_frames, ignore_index=True)
    book_df.to_parquet(OUTPUT_DIR / "book_summary.parquet")

    trade_df = build_trade_summary(trade_events)
    trade_df.to_parquet(OUTPUT_DIR / "trade_summary.parquet")

    print(f"\nbook_summary: {len(book_df)} rows")
    print(book_df.describe())
    print(f"\ntrade_summary: {len(trade_df)} rows")
    print(trade_df.describe())
    print(book_df[["spread", "best_bid_px", "best_ask_px"]].describe())


if __name__ == "__main__":
    main()