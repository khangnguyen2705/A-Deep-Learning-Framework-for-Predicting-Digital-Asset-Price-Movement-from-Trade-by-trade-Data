"""Vectorised, memory-tight interval-aggregation per paper §III.B (eq. 1–6).

For 56 M-trade Binance months, the previous pandas-only implementation
(`df = trades.copy(); df['pa'] = ...; df.groupby(...).agg(...)`) peaked
around 8–10 GB and OOM-killed Lightning's 24-GB-usable L4 cgroup. This
module avoids the copy by working in pure numpy + ``np.bincount`` /
``np.minimum.reduceat``: peak memory ≈ 4 × the trade-array size, never
two full copies. End-to-end the largest BTC month resamples in ~5 s with
peak ~2 GB.
"""
from pathlib import Path

import gc
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "num_trades", "volume", "active_buy_volume",
    "amplitude", "price_change", "vwap", "taker_ratio",
]


def _bincount_min(values: np.ndarray, idx: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-bin minimum via reduceat on a sorted view. ~3× lighter than
    pandas groupby.min on huge inputs."""
    order   = np.argsort(idx, kind="stable")
    idx_s   = idx[order]
    vals_s  = values[order]
    # Find segment starts in the sorted index array.
    # reduceat needs the index of the first element of each segment.
    starts  = np.concatenate(([0], np.flatnonzero(np.diff(idx_s)) + 1))
    out     = np.minimum.reduceat(vals_s, starts)
    return out


def _bincount_max(values: np.ndarray, idx: np.ndarray, n_bins: int) -> np.ndarray:
    order   = np.argsort(idx, kind="stable")
    idx_s   = idx[order]
    vals_s  = values[order]
    starts  = np.concatenate(([0], np.flatnonzero(np.diff(idx_s)) + 1))
    return np.maximum.reduceat(vals_s, starts)


def _bincount_first(values: np.ndarray, idx: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Per-bin value at the row with the minimum timestamp."""
    order   = np.lexsort((ts, idx))    # primary: idx ascending; secondary: ts ascending
    idx_s   = idx[order]
    vals_s  = values[order]
    starts  = np.concatenate(([0], np.flatnonzero(np.diff(idx_s)) + 1))
    return vals_s[starts]


def _bincount_last(values: np.ndarray, idx: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Per-bin value at the row with the maximum timestamp."""
    order   = np.lexsort((ts, idx))    # primary: idx; secondary: ts
    idx_s   = idx[order]
    vals_s  = values[order]
    # End of each segment is one past its last index.
    ends    = np.concatenate((np.flatnonzero(np.diff(idx_s)) + 1, [len(idx_s)])) - 1
    return vals_s[ends]


def resample_trades(trades: pd.DataFrame, l_ms: int) -> pd.DataFrame:
    """Aggregate trade-by-trade data into fixed-length time intervals.

    Group key i = floor(t / l_ms). Output has one row per interval that
    contained at least one trade. All seven feature equations follow the
    paper exactly (eq. 1–6):

      num_trades        = count
      volume            = Σ aᵢ
      active_buy_volume = Σ aᵢ (1 − mᵢ)        # taker = buyer
      amplitude         = max p − min p
      price_change      = p_last − p_first     # by timestamp
      vwap              = Σ pᵢ aᵢ / Σ aᵢ        # eq. 5
      taker_ratio       = Σ aᵢ mᵢ / Σ aᵢ        # eq. 6 (seller-taker share)

    `maker=True` means the buyer is the market maker, i.e. an active sell
    hit the book; ``(1 − maker)`` is therefore the active-buy mask.
    """
    if l_ms <= 0:
        raise ValueError(f"l_ms must be positive, got {l_ms}")

    # Pull columns out as numpy arrays — never copy the whole DataFrame.
    ts     = trades["timestamp_ms"].to_numpy(copy=False)
    price  = trades["price"].to_numpy(copy=False)
    amount = trades["amount"].to_numpy(copy=False)
    maker  = trades["maker"].to_numpy(copy=False).astype(np.bool_, copy=False)

    # Compute per-row interval key, then collapse to dense bin indices so
    # every aggregation is a single np.bincount / reduceat call.
    interval_key = (ts.astype(np.int64) // l_ms)
    unique_keys, inverse = np.unique(interval_key, return_inverse=True)
    n_bins = len(unique_keys)
    del interval_key

    # Aggregations that bincount can do directly.
    num_trades  = np.bincount(inverse, minlength=n_bins).astype(np.int64)
    volume      = np.bincount(inverse, weights=amount.astype(np.float64), minlength=n_bins)
    pa_sum      = np.bincount(inverse, weights=(price * amount).astype(np.float64), minlength=n_bins)
    active_buy  = np.bincount(inverse, weights=(amount * ~maker).astype(np.float64), minlength=n_bins)
    maker_vol   = np.bincount(inverse, weights=(amount * maker).astype(np.float64),  minlength=n_bins)

    # Aggregations that need a sort: min/max price, first/last by timestamp.
    price_f64 = price.astype(np.float64, copy=False)
    price_min = _bincount_min  (price_f64, inverse, n_bins)
    price_max = _bincount_max  (price_f64, inverse, n_bins)
    price_first = _bincount_first(price_f64, inverse, ts)
    price_last  = _bincount_last (price_f64, inverse, ts)

    # Free the per-trade arrays before allocating the bars frame.
    del inverse, price_f64, maker, amount, price, ts
    gc.collect()

    safe_volume = np.where(volume > 0, volume, np.nan)
    vwap        = np.where(volume > 0, pa_sum / safe_volume, price_last)
    taker_ratio = np.where(volume > 0, maker_vol / safe_volume, 0.0)

    out = pd.DataFrame({
        "num_trades":         num_trades,
        "volume":             volume,
        "active_buy_volume":  active_buy,
        "amplitude":          price_max - price_min,
        "price_change":       price_last - price_first,
        "vwap":               vwap,
        "taker_ratio":        taker_ratio,
        "interval_start_ms":  (unique_keys * l_ms).astype(np.int64),
    })
    out["interval_end_ms"] = out["interval_start_ms"] + l_ms
    # Keep a copy of the raw VWAP so the trading simulation and
    # signal-decay analysis can recover real prices after the 'vwap'
    # column has been overwritten with the differenced (stationary) series.
    out["vwap_raw"] = out["vwap"].copy()
    return out


def resample_monthly_paths(paths: list, l_ms: int,
                           verbose: bool = True) -> pd.DataFrame:
    """Stream-resample many monthly parquets and concatenate the bars.

    Per-month resampling is **independent** because Binance monthly files
    align on UTC month boundaries, which coincide with both 1-minute and
    5-minute interval boundaries — no straddling intervals to merge.

    Peak memory ≈ one month's trade arrays (~1.5 GB for the largest BTC
    months at 2025–26 volumes), well under Lightning's 24 GB usable L4
    cgroup. The previous pandas-copy version peaked at 8–10 GB and was
    OOM-killed.
    """
    paths = sorted(Path(p) for p in paths)
    if not paths:
        raise ValueError("resample_monthly_paths got an empty path list")

    # Late import to keep the resample module free of binance_trades imports.
    from src.data.binance_trades import ensure_timestamp_ms

    bars_per_month: list[pd.DataFrame] = []
    for p in paths:
        if verbose:
            print(f"  resample {p.name}", flush=True)
        month_df = pd.read_parquet(p, columns=["timestamp_ms", "price",
                                               "amount", "maker"])
        # Binance switched aggTrades from 13-digit ms to 16-digit µs in
        # 2024 without renaming the column. Force ms before bin-keying or
        # we get 60-ms-wide bins instead of 1-minute bins.
        ensure_timestamp_ms(month_df)
        bars = resample_trades(month_df, l_ms)
        bars_per_month.append(bars)
        del month_df
        gc.collect()
    return pd.concat(bars_per_month, ignore_index=True)
