import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import zipfile
import io
import time
import json
import concurrent.futures


BASE_URL = "https://data.binance.vision/data/spot/monthly/aggTrades"
DAILY_URL = "https://data.binance.vision/data/spot/daily/aggTrades"


def url_for(symbol: str, year: int, month: int, daily: bool = False) -> str:
    base = DAILY_URL if daily else BASE_URL
    prefix = "daily" if daily else "monthly"
    ym = f"{year}-{month:02d}"
    filename = f"{symbol.upper()}-aggTrades-{ym}.zip"
    return f"{base}/{symbol.upper()}/{filename}"


def parse_aggtrade_csv(content: str) -> pd.DataFrame:
    rows = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        rows.append({
            "timestamp_ms": int(parts[5]),
            "price": float(parts[1]),
            "amount": float(parts[2]),
            "maker": parts[6].strip().lower() == "true",
        })
    return pd.DataFrame(rows)


def download_month(symbol: str, year: int, month: int, data_dir: str,
                   daily: bool = False) -> pd.DataFrame:
    path = Path(data_dir) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    cached = path / f"{symbol.upper()}-aggTrades-{year}-{month:02d}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    url = url_for(symbol, year, month, daily=daily)
    print(f"  Downloading {url} ...")
    try:
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
    except Exception as e:
        print(f"  FAILED ({e})")
        return pd.DataFrame()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            content = f.read().decode("utf-8")

    df = parse_aggtrade_csv(content)
    if len(df) > 0:
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        df.to_parquet(cached)
        print(f"  Saved {len(df):,} trades -> {cached}")
    return df


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    months = []
    cur = datetime(s.year, s.month, 1)
    while cur <= e:
        months.append((cur.year, cur.month))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def download_range(symbol: str, start_date: str, end_date: str,
                   data_dir: str, max_workers: int = 4) -> pd.DataFrame:
    months = month_range(start_date, end_date)
    print(f"Fetching {len(months)} months of {symbol} aggTrades ({start_date} to {end_date})")
    dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(download_month, symbol, y, m, data_dir): (y, m)
            for y, m in months
        }
        for fut in concurrent.futures.as_completed(futures):
            ym = futures[fut]
            try:
                df = fut.result()
                if len(df) > 0:
                    dfs.append(df)
                    print(f"  {ym[0]}-{ym[1]:02d}: {len(df):,} trades")
            except Exception as e:
                print(f"  {ym[0]}-{ym[1]:02d}: ERROR {e}")

    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values("timestamp_ms").reset_index(drop=True)
    return result


def fetch_btc_usdt(data_dir: str, force: bool = False):
    data_dir = Path(data_dir)
    train_path = data_dir / "btc_usdt_train.parquet"
    test_path = data_dir / "btc_usdt_test.parquet"

    if not force and train_path.exists() and test_path.exists():
        print("BTC-USDT data already exists, skipping download.")
        return

    print("\n=== BTC-USDT Training Data (2019-01 to 2019-11) ===")
    train = download_range("BTCUSDT", "2019-01-01", "2019-11-30", str(data_dir))
    if len(train) > 0:
        train.to_parquet(train_path)
        print(f"Training data: {len(train):,} trades -> {train_path}")

    print("\n=== BTC-USDT Test Data (2019-12 to 2020-02) ===")
    test = download_range("BTCUSDT", "2019-12-01", "2020-02-28", str(data_dir))
    if len(test) > 0:
        test.to_parquet(test_path)
        print(f"Test data: {len(test):,} trades -> {test_path}")


def fetch_other_pairs(data_dir: str, symbols: list[str] = None, force: bool = False):
    if symbols is None:
        symbols = ["ETHUSDT", "BCHUSDT", "LTCUSDT", "EOSUSDT"]
    data_dir = Path(data_dir)

    display_names = {"ETHUSDT": "ETH", "BCHUSDT": "BCH", "LTCUSDT": "LTC", "EOSUSDT": "EOS"}
    for sym in symbols:
        out_name = display_names.get(sym, sym.replace("USDT", ""))
        out_path = data_dir / f"{out_name}_test.parquet"
        if not force and out_path.exists():
            print(f"{out_name} data already exists, skipping.")
            continue
        print(f"\n=== {out_name} Data (2019-12 to 2020-02) ===")
        df = download_range(sym, "2019-12-01", "2020-02-28", str(data_dir))
        if len(df) > 0:
            df.to_parquet(out_path)
            print(f"{out_name}: {len(df):,} trades -> {out_path}")


def fetch_all(data_dir: str = "data", force: bool = False):
    fetch_btc_usdt(data_dir, force=force)
    fetch_other_pairs(data_dir, force=force)
    print("\nDone. All data downloaded to", data_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fetch_all(args.data_dir, force=args.force)
