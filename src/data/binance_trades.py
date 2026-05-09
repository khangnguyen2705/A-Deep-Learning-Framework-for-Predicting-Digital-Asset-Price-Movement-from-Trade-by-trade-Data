"""Trade-data loaders.

Two loading modes:

1. **In-memory** (``load_btc_usdt``, ``load_other_pair``) — reads the giant
   concatenated parquet into one DataFrame. Fine for the bundled 1-month
   demo (≈6 M trades, < 1 GB) but **OOMs on the full 11-month grid**
   (≈373 M trades, ≈12 GB compressed → 25–40 GB after pandas inflate).

2. **Path-based / streaming** (``btc_train_paths`` /
   ``resample_monthly_paths``) — returns the per-month parquet paths from
   ``data/raw/`` so the resampler can stream through them one month at a
   time. Peak memory ≈ one month (~1 GB).

Run.py prefers the streaming path when ``data/raw/`` is populated (which
it always is after a fetch); the in-memory loaders are kept for backward
compatibility with the demo and unit tests.
"""
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# In-memory loaders (small datasets only)
# ---------------------------------------------------------------------------

def load_trades(filepath: str) -> pd.DataFrame:
    df = pd.read_parquet(filepath)
    required = {"timestamp_ms", "price", "amount", "maker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {filepath}")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def load_btc_usdt(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = load_trades(str(data_dir / "btc_usdt_train.parquet"))
    test  = load_trades(str(data_dir / "btc_usdt_test.parquet"))
    return train, test


def load_other_pair(data_dir: str, symbol: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{symbol}_test.parquet"
    return load_trades(str(path))


# ---------------------------------------------------------------------------
# Path-based loaders (streaming through ``data/raw/`` monthlies)
# ---------------------------------------------------------------------------

def _find_monthly_paths(data_dir: Path, symbol_with_quote: str,
                        year_months: list[tuple[int, int]]) -> list[Path]:
    """Return existing raw monthly parquets for ``(year, month)`` tuples."""
    raw = Path(data_dir) / "raw"
    out: list[Path] = []
    for y, m in year_months:
        p = raw / f"{symbol_with_quote}-aggTrades-{y}-{m:02d}.parquet"
        if p.exists() and p.stat().st_size > 0:
            out.append(p)
    return sorted(out)


def btc_train_paths(data_dir: str) -> list[Path]:
    from src.data.fetch_binance import TRAIN_START, TRAIN_END, month_range
    return _find_monthly_paths(
        Path(data_dir), "BTCUSDT", month_range(TRAIN_START, TRAIN_END))


def btc_test_paths(data_dir: str) -> list[Path]:
    from src.data.fetch_binance import TEST_START, TEST_END, month_range
    return _find_monthly_paths(
        Path(data_dir), "BTCUSDT", month_range(TEST_START, TEST_END))


def other_pair_paths(data_dir: str, symbol: str) -> list[Path]:
    """Return the test-window monthly parquets for an altcoin transfer pair.

    Accepts either ``"ETH"`` (display name) or ``"ETHUSDT"`` (full symbol).
    """
    from src.data.fetch_binance import TEST_START, TEST_END, month_range
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    return _find_monthly_paths(
        Path(data_dir), sym, month_range(TEST_START, TEST_END))
