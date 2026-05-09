"""Memory-efficient Binance aggTrades fetcher.

Designed to run on a 16 GB CPU box: each monthly download is parsed straight
from the zip stream into a pandas DataFrame, written to a per-month parquet
file under ``data/raw/``, and then dropped from memory. The final train /
test parquets are produced by streaming the monthly parquet files through
``pyarrow.ParquetWriter`` so we never hold more than one month at a time.
"""
import concurrent.futures
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


BASE_URL  = "https://data.binance.vision/data/spot/monthly/aggTrades"
DAILY_URL = "https://data.binance.vision/data/spot/daily/aggTrades"

# aggTrades CSV columns (Binance spec):
# 0 a (agg trade id), 1 p (price), 2 q (qty), 3 f (first id), 4 l (last id),
# 5 T (timestamp ms), 6 m (maker), 7 M (best price match)
_AGGTRADE_COLS  = ["a", "price", "amount", "f", "l", "timestamp_ms", "maker", "M"]
_AGGTRADE_KEEP  = ["timestamp_ms", "price", "amount", "maker"]
_AGGTRADE_DTYPE = {
    "timestamp_ms": "int64",
    "price":        "float64",
    "amount":       "float64",
    "maker":        "boolean",   # nullable bool to tolerate odd rows
}


def url_for(symbol: str, year: int, month: int, daily: bool = False) -> str:
    base = DAILY_URL if daily else BASE_URL
    ym = f"{year}-{month:02d}"
    filename = f"{symbol.upper()}-aggTrades-{ym}.zip"
    return f"{base}/{symbol.upper()}/{filename}"


def _parse_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Parse Binance aggTrades CSV bytes into a slim DataFrame.

    Avoids materialising a Python list-of-dicts (which is what blew up the
    previous implementation on 16 GB instances). pandas' C-level parser is
    both faster and ~10× lighter on memory.
    """
    df = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        names=_AGGTRADE_COLS,
        usecols=_AGGTRADE_KEEP,
        dtype=_AGGTRADE_DTYPE,
        engine="c",
    )
    df["maker"] = df["maker"].astype(bool)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def download_month(symbol: str, year: int, month: int, data_dir: str,
                   daily: bool = False) -> Path:
    """Download one month, write a parquet, return the cached path.

    Returns the path even when the parquet is already cached (idempotent).
    Returns ``None`` on hard download failure so the caller can decide.
    """
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached = raw_dir / f"{symbol.upper()}-aggTrades-{year}-{month:02d}.parquet"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    url = url_for(symbol, year, month, daily=daily)
    print(f"  Downloading {url} ...", flush=True)
    try:
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
    except Exception as e:
        print(f"  FAILED ({e})", flush=True)
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            csv_bytes = f.read()
    del resp                           # free the HTTP buffer ASAP

    df = _parse_csv_bytes(csv_bytes)
    del csv_bytes
    if len(df) == 0:
        print(f"  WARNING: empty parse for {url}", flush=True)
        return None
    df.to_parquet(cached, compression="snappy", index=False)
    print(f"  Saved {len(df):,} trades -> {cached}", flush=True)
    return cached


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    months, cur = [], datetime(s.year, s.month, 1)
    while cur <= e:
        months.append((cur.year, cur.month))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def download_range(symbol: str, start_date: str, end_date: str,
                   data_dir: str, max_workers: int = 2) -> list[Path]:
    """Download every month in [start_date, end_date], return cached paths.

    ``max_workers`` defaults to 2 so two months are unzipped + parsed
    simultaneously — 4 parallel parses overran 16 GB on the original code.
    """
    months = month_range(start_date, end_date)
    print(
        f"Fetching {len(months)} months of {symbol} aggTrades "
        f"({start_date} to {end_date}) — max_workers={max_workers}",
        flush=True,
    )
    paths: list[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(download_month, symbol, y, m, data_dir): (y, m)
            for y, m in months
        }
        for fut in concurrent.futures.as_completed(futures):
            ym = futures[fut]
            try:
                p = fut.result()
            except Exception as e:
                print(f"  {ym[0]}-{ym[1]:02d}: ERROR {e}", flush=True)
                continue
            if p is not None:
                paths.append(p)
    paths.sort()                       # chronological
    return paths


def _stream_concat(paths: list[Path], out_path: Path) -> int:
    """Stream-concat monthly parquets into one file via pyarrow.

    Holds at most one month in memory at any time. Returns total row count.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    writer = None
    total = 0
    for p in paths:
        table = pq.read_table(p)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)
        total += table.num_rows
        del table
    if writer is not None:
        writer.close()
    return total


# ---------------------------------------------------------------------------
# Public scheduling API
# ---------------------------------------------------------------------------

# Date schedule for the paper-equivalent run on contemporary data:
#   Train: 2025-03-01 → 2026-01-31 (11 months — matches paper's training span)
#   Test : 2026-02-01 → 2026-04-30 (3 months  — matches paper's test span)
# data.binance.vision publishes the prior month's aggTrades a day or two into
# the next month, so this schedule is fully fetchable as of mid-May 2026.
TRAIN_START = "2025-03-01"
TRAIN_END   = "2026-01-31"
TEST_START  = "2026-02-01"
TEST_END    = "2026-04-30"


def fetch_btc_usdt(data_dir: str, force: bool = False, max_workers: int = 2):
    data_dir = Path(data_dir)
    train_path = data_dir / "btc_usdt_train.parquet"
    test_path  = data_dir / "btc_usdt_test.parquet"

    if not force and train_path.exists() and test_path.exists():
        print("BTC-USDT data already exists, skipping download.")
        return

    print(f"\n=== BTC-USDT Training Data ({TRAIN_START} to {TRAIN_END}) ===")
    train_paths = download_range(
        "BTCUSDT", TRAIN_START, TRAIN_END, str(data_dir),
        max_workers=max_workers,
    )
    if train_paths:
        n = _stream_concat(train_paths, train_path)
        print(f"Training data: {n:,} trades -> {train_path}")

    print(f"\n=== BTC-USDT Test Data ({TEST_START} to {TEST_END}) ===")
    test_paths = download_range(
        "BTCUSDT", TEST_START, TEST_END, str(data_dir),
        max_workers=max_workers,
    )
    if test_paths:
        n = _stream_concat(test_paths, test_path)
        print(f"Test data: {n:,} trades -> {test_path}")


def fetch_other_pairs(data_dir: str, symbols: list[str] = None,
                      force: bool = False, max_workers: int = 2):
    if symbols is None:
        symbols = ["ETHUSDT", "BCHUSDT", "LTCUSDT", "EOSUSDT"]
    data_dir = Path(data_dir)
    display_names = {"ETHUSDT": "ETH", "BCHUSDT": "BCH",
                     "LTCUSDT": "LTC", "EOSUSDT": "EOS"}
    for sym in symbols:
        out_name = display_names.get(sym, sym.replace("USDT", ""))
        out_path = data_dir / f"{out_name}_test.parquet"
        if not force and out_path.exists():
            print(f"{out_name} data already exists, skipping.")
            continue
        print(f"\n=== {out_name} Data ({TEST_START} to {TEST_END}) ===")
        paths = download_range(
            sym, TEST_START, TEST_END, str(data_dir),
            max_workers=max_workers,
        )
        if paths:
            n = _stream_concat(paths, out_path)
            print(f"{out_name}: {n:,} trades -> {out_path}")


def fetch_all(data_dir: str = "data", force: bool = False, max_workers: int = 2):
    fetch_btc_usdt(data_dir, force=force, max_workers=max_workers)
    fetch_other_pairs(data_dir, force=force, max_workers=max_workers)
    print("\nDone. All data downloaded to", data_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-workers", type=int, default=2,
        help="Parallel monthly downloads (default 2). Lower to 1 if you "
             "OOM on a small CPU instance; raise to 4 on 64+ GB boxes.",
    )
    args = parser.parse_args()
    fetch_all(args.data_dir, force=args.force, max_workers=args.max_workers)
