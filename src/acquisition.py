import json
import time
import threading
from pathlib import Path

import requests
import websocket

SYMBOL = "btcusdt"
REST_SYMBOL = SYMBOL.upper()
SNAPSHOT_LIMIT = 5000
RUN_DURATION_SECONDS = 9 * 60 * 60

STREAM_URL = (
    f"wss://stream.binance.us:9443/stream?streams={SYMBOL}@depth@100ms/{SYMBOL}@trade"
)
REST_SNAPSHOT_URL = (
    f"https://api.binance.us/api/v3/depth?symbol={REST_SYMBOL}&limit={SNAPSHOT_LIMIT}"
)

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

depth_file = open(OUTPUT_DIR / "depth_events.jsonl", "a")
trade_file = open(OUTPUT_DIR / "trade_events.jsonl", "a")
snapshot_file = open(OUTPUT_DIR / "snapshots.jsonl", "a")

segment_id = 0
lock = threading.Lock()
stop_requested = threading.Event()
start_time = time.time()


def fetch_snapshot():
    global segment_id
    resp = requests.get(REST_SNAPSHOT_URL, timeout=10)
    resp.raise_for_status()
    snapshot = resp.json()

    with lock:
        segment_id += 1
        current_segment = segment_id

    record = {
        "segment_id": current_segment,
        "local_time": time.time(),
        "snapshot": snapshot,
    }
    snapshot_file.write(json.dumps(record) + "\n")
    snapshot_file.flush()
    print(f"segment {current_segment}: snapshot taken, lastUpdateId={snapshot['lastUpdateId']}")


def on_open(ws):
    fetch_snapshot()


def on_message(ws, message):
    payload = json.loads(message)
    stream = payload.get("stream", "")
    data = payload.get("data", {})

    record = {
        "segment_id": segment_id,
        "local_time": time.time(),
        "data": data,
    }

    if stream.endswith("@depth@100ms"):
        depth_file.write(json.dumps(record) + "\n")
        depth_file.flush()
    elif stream.endswith("@trade"):
        trade_file.write(json.dumps(record) + "\n")
        trade_file.flush()


def on_error(ws, error):
    print(f"websocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    print("websocket connection closed")


def watchdog(ws):
    remaining = RUN_DURATION_SECONDS - (time.time() - start_time)
    if remaining > 0:
        time.sleep(remaining)
    stop_requested.set()
    try:
        ws.close()
    except Exception:
        pass


def run():
    while not stop_requested.is_set():
        remaining = RUN_DURATION_SECONDS - (time.time() - start_time)
        if remaining <= 0:
            stop_requested.set()
            break

        ws = websocket.WebSocketApp(
            STREAM_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        watchdog_thread = threading.Thread(target=watchdog, args=(ws,), daemon=True)
        watchdog_thread.start()

        ws.run_forever()

        if not stop_requested.is_set():
            print("websocket closed unexpectedly, reconnecting in 5s...")
            time.sleep(5)

    depth_file.close()
    trade_file.close()
    snapshot_file.close()
    print(f"acquisition finished after {(time.time() - start_time) / 3600:.2f} hours, files closed.")


if __name__ == "__main__":
    run()