"""Quant metrics beyond the paper's reported accuracy and loss.

Modules:
  - Information Coefficient (IC) and rolling IC
  - Binomial significance test for directional accuracy
  - Signal decay: accuracy vs hold horizon
  - Regime accuracy split (high-vol / low-vol)

All functions are pure NumPy / pandas / scipy; no TF dependency.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------

def compute_ic(prob_up: np.ndarray, realized_cm: np.ndarray) -> dict:
    """Spearman rank IC between P(UP) and realised C(m).

    IC measures signal quality as a continuous variable rather than the
    binary accuracy metric. An IC of ~0.02–0.05 is considered tradeable
    at Jane Street's scale.

    Args:
        prob_up:      (N,) predicted probability of UP move.
        realized_cm:  (N,) realised C(m) return (not the label, the raw return).

    Returns:
        Dict with ic, ic_t_stat, ic_pval.
    """
    mask = np.isfinite(prob_up) & np.isfinite(realized_cm)
    if mask.sum() < 10:
        return {"ic": None, "ic_t_stat": None, "ic_pval": None}

    ic, pval = spearmanr(prob_up[mask], realized_cm[mask])
    n = mask.sum()
    t_stat = ic * np.sqrt((n - 2) / (1 - ic ** 2 + 1e-12))
    return {
        "ic":        round(float(ic), 6),
        "ic_t_stat": round(float(t_stat), 4),
        "ic_pval":   round(float(pval), 6),
    }


def compute_rolling_ic(
    prob_up: np.ndarray,
    realized_cm: np.ndarray,
    timestamps_ms: np.ndarray,
    window: str = "7D",
) -> pd.DataFrame:
    """Rolling IC in a calendar window (e.g. '7D' = 7-day window).

    Returns DataFrame with columns: window_start, ic, n_obs.
    """
    df = pd.DataFrame(
        {"prob_up": prob_up, "realized_cm": realized_cm},
        index=pd.to_datetime(timestamps_ms, unit="ms", utc=True),
    ).dropna()

    records = []
    for key, grp in df.groupby(pd.Grouper(freq=window)):
        if len(grp) < 5:
            continue
        ic_val, _ = spearmanr(grp["prob_up"], grp["realized_cm"])
        records.append({
            "window_start": str(key),
            "ic":           round(float(ic_val), 6),
            "n_obs":        int(len(grp)),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Statistical Significance
# ---------------------------------------------------------------------------

def accuracy_significance(
    n_correct: int,
    n_total: int,
    null_p: float = 0.5,
) -> dict:
    """One-sided binomial test: is accuracy > null_p (50%) significant?

    Args:
        n_correct:  Number of correct directional predictions.
        n_total:    Total predictions.
        null_p:     Null hypothesis accuracy (default 0.5 = random).

    Returns:
        Dict with accuracy, z_score, p_value, significant_1pct, significant_5pct.
    """
    if n_total == 0:
        return {}
    acc = n_correct / n_total
    result = binomtest(n_correct, n_total, null_p, alternative="greater")
    z = (acc - null_p) / np.sqrt(null_p * (1 - null_p) / n_total)
    return {
        "n_correct":        n_correct,
        "n_total":          n_total,
        "accuracy":         round(acc, 6),
        "z_score":          round(float(z), 4),
        "p_value":          round(float(result.pvalue), 6),
        "significant_5pct": bool(result.pvalue < 0.05),
        "significant_1pct": bool(result.pvalue < 0.01),
    }


# ---------------------------------------------------------------------------
# Signal Decay
# ---------------------------------------------------------------------------

def compute_signal_decay(
    model,
    feature_data: np.ndarray,
    vwap: np.ndarray,
    t_indices: np.ndarray,
    T: int,
    base_m: int,
    extra_horizons: int = 5,
) -> list[dict]:
    """Measure model accuracy as the label horizon extends past base_m.

    For each horizon h in {base_m, base_m+1, ..., base_m+extra_horizons},
    relabel using C(h)_t = vwap[t+h]/vwap[t]-1 (eps=0), run the model on
    the same trailing windows, and report accuracy at horizon h.

    The signal (predictions) does not change across h — only the labels do.
    A flat curve means the edge persists; a steep drop means decay.
    """
    from src.datasets.windowing import _minmax_normalize_window
    n_vwap = len(vwap)

    # Build prediction windows once (per-window minmax normalisation).
    valid_t = [t for t in t_indices if t >= T and t < len(feature_data)]
    if not valid_t:
        return []
    X = np.stack([
        _minmax_normalize_window(feature_data[t - T:t]) for t in valid_t
    ]).astype(np.float32)
    proba = model.predict(X, verbose=0)
    pred_up = (np.argmax(proba, axis=1) == 0).astype(np.int64)

    results = []
    for h in range(base_m, base_m + extra_horizons + 1):
        n_correct = 0
        n_valid = 0
        for i, t in enumerate(valid_t):
            if t + h >= n_vwap:
                continue
            ret = vwap[t + h] / vwap[t] - 1.0
            if not np.isfinite(ret):
                continue
            true_up = int(ret > 0)
            n_valid += 1
            if pred_up[i] == true_up:
                n_correct += 1
        acc = n_correct / n_valid if n_valid > 0 else float("nan")
        results.append({
            "horizon": h,
            "label_accuracy": round(float(acc), 6),
            "n_samples": n_valid,
        })
    return results


# ---------------------------------------------------------------------------
# Regime Analysis
# ---------------------------------------------------------------------------

def tag_regimes(
    intervals: pd.DataFrame,
    vol_col: str = "amplitude",
    window: int = 24,
) -> pd.DataFrame:
    """Tag each interval as 'high_vol' or 'low_vol' based on rolling amplitude std.

    Args:
        intervals:  Interval DataFrame (must have `amplitude` column).
        vol_col:    Column to use as a volatility proxy.
        window:     Rolling window in bars.

    Returns:
        intervals with a new 'regime' column added.
    """
    intervals = intervals.copy()
    rv = intervals[vol_col].rolling(window, min_periods=1).std()
    median_rv = rv.median()
    intervals["regime"] = np.where(rv >= median_rv, "high_vol", "low_vol")
    return intervals


def compute_regime_accuracy(
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    intervals: pd.DataFrame,
    t_indices: np.ndarray,
) -> dict:
    """Split accuracy by regime (high_vol / low_vol).

    Returns dict keyed by regime name with accuracy and n_samples.
    """
    if "regime" not in intervals.columns:
        raise ValueError("intervals must have a 'regime' column. Run tag_regimes first.")

    regimes = intervals["regime"].values[t_indices]
    results = {}
    for reg in np.unique(regimes):
        mask = regimes == reg
        acc = float(np.mean(y_pred_class[mask] == y_true_class[mask]))
        sig = accuracy_significance(
            int(np.sum(y_pred_class[mask] == y_true_class[mask])),
            int(np.sum(mask)),
        )
        results[reg] = {"accuracy": round(acc, 6), "n_samples": int(mask.sum()),
                        **sig}
    return results
