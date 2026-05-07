"""Stationarity testing and fixing for interval feature arrays.

The paper (Section III.C) runs ADF on all 7 features and finds only the
vwap (price, eq. 5) series is non-stationary (stat=-1.454, p=0.55).
It applies first-differencing to vwap: price'(t) = price(t) - price(t-1).

We replicate that exactly: run ADF for logging purposes, then unconditionally
difference vwap and drop the first row (which becomes NaN after diff).
"""
import numpy as np
import json
from statsmodels.tsa.stattools import adfuller

FEATURE_NAMES = [
    "num_trades", "volume", "active_buy_volume",
    "amplitude", "price_change", "vwap", "taker_ratio",
]
VWAP_IDX = 5          # index of vwap in the feature array


def run_adf_report(features: np.ndarray,
                   names: list = None) -> dict:
    """Run ADF on every feature column and return a stats dict for logging."""
    if names is None:
        names = FEATURE_NAMES
    report = {}
    for i, name in enumerate(names):
        series = features[:, i]
        valid = series[~np.isnan(series)]
        if len(valid) < 20:
            report[name] = {"skipped": True}
            continue
        try:
            stat, pval, _, _, crit, _ = adfuller(valid)
            report[name] = {
                "adf_stat": round(float(stat), 4),
                "p_value": round(float(pval), 4),
                "stationary_10pct": bool(pval < 0.10),
                "critical_10pct": round(float(crit["10%"]), 4),
            }
        except Exception as e:
            report[name] = {"error": str(e)}
    return report


def apply_vwap_differencing(features: np.ndarray) -> np.ndarray:
    """Difference the vwap column in-place and drop the first (NaN) row.

    This mirrors paper equation (8): price'(t) = price(t) - price(t-1).
    Returns the array with row 0 removed (shape goes from N to N-1).
    """
    features = features.copy()
    # np.diff produces N-1 values; prepend NaN so indices align
    features[:, VWAP_IDX] = np.concatenate(
        [[np.nan], np.diff(features[:, VWAP_IDX])]
    )
    # Drop the first row which is NaN after differencing
    return features[1:]


def prepare_features(features: np.ndarray,
                     report_path: str = None) -> np.ndarray:
    """Run ADF report (optional save), then apply vwap differencing.

    Args:
        features:     Shape (N, 7) float array from resample_trades.
        report_path:  If given, write ADF stats JSON to this path.

    Returns:
        Shape (N-1, 7) array with vwap differenced and first row dropped.
    """
    report = run_adf_report(features)
    if report_path:
        import pathlib
        pathlib.Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    return apply_vwap_differencing(features)

