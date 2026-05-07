import numpy as np
import pandas as pd


def compute_Cm(intervals: pd.DataFrame, m: int, l_ms: int) -> np.ndarray:
    vwap = intervals["vwap"].values
    n = len(vwap)
    Cm = np.full(n, np.nan)
    m_ahead = np.arange(m, n)
    lag = np.arange(n - m)
    Cm[lag] = vwap[m_ahead] / vwap[lag] - 1.0
    return Cm


def make_labels(Cm: np.ndarray, epsilon: float) -> np.ndarray:
    mask = Cm > epsilon
    labels = np.zeros((len(Cm), 2), dtype=np.float32)
    labels[mask, 0] = 1.0
    labels[~mask, 1] = 1.0
    return labels
