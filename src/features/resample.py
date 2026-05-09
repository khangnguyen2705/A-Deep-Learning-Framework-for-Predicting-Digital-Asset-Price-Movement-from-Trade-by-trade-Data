"""Vectorised interval-aggregation per paper Section III.B equations (1)-(6)."""
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "num_trades", "volume", "active_buy_volume",
    "amplitude", "price_change", "vwap", "taker_ratio",
]


def resample_trades(trades: pd.DataFrame, l_ms: int) -> pd.DataFrame:
    """Aggregate trade-by-trade data into fixed-length time intervals.

    Group key i = floor(t / l_ms). Output has one row per interval that
    contained at least one trade. All seven feature equations follow the
    paper exactly:

      num_trades        = count
      volume            = Σ aᵢ
      active_buy_volume = Σ aᵢ (1 − mᵢ)        # taker = buyer
      amplitude         = max p − min p
      price_change      = p_last − p_first
      vwap              = Σ pᵢ aᵢ / Σ aᵢ        # eq. 5
      taker_ratio       = Σ aᵢ mᵢ / Σ aᵢ        # eq. 6 (seller-taker share)

    `maker=True` means the buyer is the market maker, i.e. an active sell hit
    the book. (1 − maker) is therefore the active-buy mask.

    Implementation note: for ~6 M trades a Python `for key, group in groupby`
    loop takes ~10 minutes. The groupby.agg path below is fully vectorised
    and runs in a few seconds.
    """
    if l_ms <= 0:
        raise ValueError(f"l_ms must be positive, got {l_ms}")

    df = trades.loc[:, ["timestamp_ms", "price", "amount", "maker"]].copy()
    df["interval_key"] = (df["timestamp_ms"] // l_ms).astype(np.int64)
    maker = df["maker"].astype(np.float64).to_numpy()
    df["pa"]            = df["price"] * df["amount"]
    df["active_buy"]    = df["amount"] * (1.0 - maker)
    df["maker_amount"]  = df["amount"] * maker

    grouped = df.groupby("interval_key", sort=True)
    agg = grouped.agg(
        num_trades        =("price",        "size"),
        volume            =("amount",       "sum"),
        active_buy_volume =("active_buy",   "sum"),
        maker_volume      =("maker_amount", "sum"),
        pa_sum            =("pa",           "sum"),
        price_min         =("price",        "min"),
        price_max         =("price",        "max"),
        price_first       =("price",        "first"),
        price_last        =("price",        "last"),
    ).reset_index()

    safe_volume = agg["volume"].where(agg["volume"] > 0, np.nan)
    vwap = (agg["pa_sum"] / safe_volume).fillna(agg["price_last"])
    taker_ratio = (agg["maker_volume"] / safe_volume).fillna(0.0)

    out = pd.DataFrame({
        "num_trades":         agg["num_trades"].astype(np.int64),
        "volume":             agg["volume"].astype(np.float64),
        "active_buy_volume":  agg["active_buy_volume"].astype(np.float64),
        "amplitude":          (agg["price_max"] - agg["price_min"]).astype(np.float64),
        "price_change":       (agg["price_last"] - agg["price_first"]).astype(np.float64),
        "vwap":               vwap.astype(np.float64),
        "taker_ratio":        taker_ratio.astype(np.float64),
        "interval_start_ms":  (agg["interval_key"] * l_ms).astype(np.int64),
    })
    out["interval_end_ms"] = out["interval_start_ms"] + l_ms
    # Keep a copy of the raw VWAP so downstream code (trading simulation,
    # signal-decay analysis) can recover real prices after the 'vwap' column
    # has been overwritten with the differenced (stationary) series.
    out["vwap_raw"] = out["vwap"].copy()
    return out


def resample_monthly_paths(paths: list, l_ms: int,
                           verbose: bool = True) -> pd.DataFrame:
    """Stream-resample many monthly parquets and concatenate the bars.

    Avoids the OOM that ``resample_trades`` hits on a single concatenated
    11-month parquet (~12 GB compressed → ~25–40 GB after pandas inflation
    plus the 5 derived columns). Each month is read, resampled, and then
    freed before the next one is loaded; peak memory ≈ one month
    (~1 GB for modern BTC volume).

    Per-month resampling is **independent** because Binance monthly files
    align on UTC month boundaries, which coincide with both 1-minute and
    5-minute interval boundaries. There are no straddling intervals to
    merge across files.

    Args:
        paths:    Iterable of monthly parquet paths, in any order — they
                  are sorted internally.
        l_ms:     Interval length (60_000 or 300_000).
        verbose:  Print per-month progress.

    Returns:
        Concatenated bars DataFrame, chronological, indices reset.
    """
    paths = sorted(Path(p) for p in paths)
    if not paths:
        raise ValueError("resample_monthly_paths got an empty path list")

    bars_per_month: list[pd.DataFrame] = []
    for p in paths:
        if verbose:
            print(f"  resample {p.name}", flush=True)
        month_df = pd.read_parquet(p, columns=["timestamp_ms", "price",
                                               "amount", "maker"])
        bars = resample_trades(month_df, l_ms)
        bars_per_month.append(bars)
        del month_df
    return pd.concat(bars_per_month, ignore_index=True)
