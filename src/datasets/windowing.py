"""Trailing-window dataset construction for the paper's LSTM pipeline.

Key design decisions (from paper Section III.D):
  - Trailing window of length T: X ∈ ℝ^(T×7)
  - Min-max normalization is applied column-wise WITHIN each window
    (NOT a global scaler — that would introduce lookahead bias).
  - Label at prediction time t is derived from Cm[t] which requires
    data t+m bars ahead; windows where Cm is NaN must be excluded.
  - Offset-based redundancy reduction: 10–50 % of eligible offsets
    are randomly sampled, plus offset = (period_length mod T) for
    full coverage.
"""
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "num_trades", "volume", "active_buy_volume",
    "amplitude", "price_change", "vwap", "taker_ratio"
]


def _minmax_normalize_window(window: np.ndarray) -> np.ndarray:
    """Min-max normalize each feature column within a single window.

    Any column with zero range is left unchanged (avoids division by zero).
    Shape: (T, num_features) → (T, num_features).
    """
    col_min = window.min(axis=0)
    col_max = window.max(axis=0)
    rng = col_max - col_min
    # Avoid division by zero for constant columns
    rng_safe = np.where(rng == 0, 1.0, rng)
    return (window - col_min) / rng_safe


def build_trailing_windows(
    intervals: pd.DataFrame,
    Cm: np.ndarray,
    labels: np.ndarray,
    T: int,
    m: int,
    offset_fraction_min: float = 0.10,
    offset_fraction_max: float = 0.50,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, t_indices) trailing-window tensors.

    Args:
        intervals:  Interval DataFrame (must contain FEATURE_COLUMNS).
        Cm:         Return series C(m)_t; NaN at the last m positions.
        labels:     One-hot labels shape (N, 2).
        T:          Trailing window length (number of intervals).
        m:          Prediction horizon (used only for NaN guard check).
        offset_fraction_min/max: Fraction of T used as offset sampling range.
        rng:        NumPy random generator for reproducibility.

    Returns:
        X:         (n_samples, T, num_features) float32 normalized windows.
        y:         (n_samples, 2) float32 one-hot labels.
        t_indices: (n_samples,) int array of prediction-time indices into
                   the original intervals DataFrame.
    """
    feature_data = intervals[FEATURE_COLUMNS].values.astype(np.float32)
    n = len(feature_data)
    if rng is None:
        rng = np.random.default_rng(42)

    start_indices = _generate_offset_starts(n, T, offset_fraction_min,
                                            offset_fraction_max, rng)
    X_list, y_list, t_list = [], [], []
    for t in start_indices:
        # Guard: prediction index t must have a valid (non-NaN) label
        if t >= n or np.isnan(Cm[t]):
            continue
        window = feature_data[t - T:t]          # shape (T, 7)
        window_norm = _minmax_normalize_window(window)
        X_list.append(window_norm)
        y_list.append(labels[t])
        t_list.append(t)

    if not X_list:
        return np.array([]), np.array([]), np.array([])
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(t_list, dtype=np.int64),
    )


def _generate_offset_starts(
    n: int, T: int,
    fraction_min: float, fraction_max: float,
    rng: np.random.Generator,
) -> list[int]:
    """Generate prediction-time indices using offset-based sampling.

    Following the paper's offset strategy:
      - Base stride: T (non-overlapping)
      - Randomly sample K offsets ∈ [1, T-1] where K ∈ [10%, 50%] of T
      - Always include offset = period_length mod T for full coverage
    """
    if n <= T:
        return []
    num_non_overlap = n // T
    if num_non_overlap <= 1:
        # Tiny segment: include every valid position
        return list(range(T, n))

    fraction = rng.uniform(fraction_min, fraction_max)
    num_offsets = max(1, int(num_non_overlap * fraction))

    possible_offsets = list(range(1, T))      # exclude 0 (already the base)
    rng.shuffle(possible_offsets)
    selected_offsets = set(possible_offsets[:num_offsets])
    # Always include the "full coverage" offset
    selected_offsets.add(n % T if n % T != 0 else T - 1)

    starts = set()
    # Base stride (offset=0)
    for k in range(num_non_overlap):
        t = k * T + T
        if t < n:
            starts.add(t)
    # Shifted strides
    for offset in selected_offsets:
        for k in range(num_non_overlap):
            t = offset + k * T + T
            if t < n:
                starts.add(t)

    return sorted(starts)

