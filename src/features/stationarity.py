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


# ADF diagnostics on the full 11-month BTC train series take ~20 min per
# column with statsmodels' default `autolag='AIC'` because the lag-search
# fits ~100 OLS regressions on the (N, k) Toeplitz matrix. ADF here is
# only a diagnostic — the paper unconditionally differences VWAP regardless
# of the test outcome — so we cap maxlag and subsample large series. The
# whole stationarity step then runs in <2 s on the full 11-month frame.
_ADF_MAX_SAMPLES = 50_000
_ADF_MAXLAG     = 24      # 24 bars ≈ 24 minutes for l=60 s, 2 hours for l=300 s


def run_adf_report(features: np.ndarray,
                   names: list = None,
                   max_samples: int = _ADF_MAX_SAMPLES,
                   maxlag: int = _ADF_MAXLAG) -> dict:
    """Run ADF on every feature column and return a stats dict for logging.

    For long series (paper's full 11-month run is ~475k bars) the default
    AIC-based ``adfuller`` lag search dominates wall-clock. ADF here is
    purely diagnostic, so we (1) take a deterministic stride sample of
    at most ``max_samples`` rows and (2) fix ``maxlag`` instead of letting
    AIC scan up to 100. Both are typical practice for stationarity checks
    on long financial series and don't change the qualitative finding
    (only VWAP is non-stationary).
    """
    if names is None:
        names = FEATURE_NAMES
    n = features.shape[0]
    if n > max_samples:
        # Stride sample preserves the time structure of the series better
        # than a random sample.
        stride  = max(1, n // max_samples)
        sampled = features[::stride]
    else:
        sampled = features

    report: dict = {
        "_meta": {
            "n_total":   int(n),
            "n_sampled": int(sampled.shape[0]),
            "maxlag":    int(maxlag),
            "autolag":   None,
        }
    }
    for i, name in enumerate(names):
        series = sampled[:, i]
        valid = series[~np.isnan(series)]
        if len(valid) < 20:
            report[name] = {"skipped": True}
            continue
        try:
            # When ``autolag=None``, adfuller returns 5 values (drops icbest);
            # with autolag set it returns 6. Unpack defensively.
            res = adfuller(valid, maxlag=maxlag, autolag=None)
            stat, pval, _, _, crit = res[:5]
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

