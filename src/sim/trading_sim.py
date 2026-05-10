"""Trading simulation following paper Section V.B.

Assumptions (from paper):
  - No market impact (small order vs book)
  - No execution latency  (fill at last price of each interval)
  - Transaction cost: 0.0003 % per executed order (= 0.000003 in decimal)
  - Positions are isolated: long and short do NOT net each other
  - Each position is held for exactly `hold_periods` intervals, then closed

Strategy: at each prediction point open long if model says UP, short if DOWN.

Jane Street extensions added here:
  - compute_sim_metrics:  Sharpe, MaxDD, Calmar, win rate, avg bps per trade
  - run_fee_stress_test:  re-run P&L at multiple fee levels (retail stress)
"""
import numpy as np
import pandas as pd


# Annualisation factor: sqrt(bars per year)
# For 5-min bars: 252 trading days × 24 h × 12 bars/h = 72 576 bars/year
_ANN = {
    60_000:   np.sqrt(252 * 24 * 60),   # 1-min bars
    300_000:  np.sqrt(252 * 24 * 12),   # 5-min bars
}


def run_trading_simulation(
    model,
    X_test: np.ndarray,
    y_pred_class: np.ndarray,
    intervals: pd.DataFrame,
    t_indices: np.ndarray,
    fee: float = 0.000003,
    hold_periods: int = 5,
    m: int = 5,
) -> pd.DataFrame:
    """Run the long/short trading simulation and return a trade log.

    Args:
        model:         Trained Keras model (not used here; kept for API compat).
        X_test:        Input windows (not used here; kept for API compat).
        y_pred_class:  (N,) int predicted class: 0=UP, 1=DOWN.
        intervals:     Full interval DataFrame (must contain vwap, interval_end_ms).
        t_indices:     (N,) int indices into `intervals` for each prediction point.
        fee:           Fee per order in decimal (0.000003 = 0.0003 %).
        hold_periods:  Number of intervals to hold each position.
        m:             Prediction horizon (informational only).

    Returns:
        DataFrame with one row per trade and a cumulative_return column.
    """
    # Prefer the raw VWAP for entry/exit prices. The 'vwap' column is the
    # differenced (stationary) series after stationarity.prepare_features
    # has been applied, so it is NOT a valid price series for filling trades.
    if "vwap_raw" in intervals.columns:
        vwap = intervals["vwap_raw"].values
    else:
        vwap = intervals["vwap"].values
    timestamps = intervals["interval_end_ms"].values

    positions = []
    n = len(t_indices)
    for i in range(n):
        exit_idx = t_indices[i] + hold_periods
        if exit_idx >= len(vwap):
            break
        direction = 1 if y_pred_class[i] == 0 else -1   # 0=UP → long
        entry_price = vwap[t_indices[i]]
        exit_price  = vwap[exit_idx]

        if entry_price <= 0 or exit_price <= 0:
            continue

        gross_return = (exit_price / entry_price - 1.0) * direction
        net_return   = gross_return - fee - fee   # entry + exit cost

        positions.append({
            "entry_time":    int(timestamps[t_indices[i]]),
            "exit_time":     int(timestamps[exit_idx]),
            "direction":     "long" if direction == 1 else "short",
            "entry_price":   float(entry_price),
            "exit_price":    float(exit_price),
            "gross_return":  float(gross_return),
            "fee_paid":      float(fee * 2),
            "net_return":    float(net_return),
        })

    if not positions:
        return pd.DataFrame()

    result = pd.DataFrame(positions)
    result["cumulative_return"] = (1.0 + result["net_return"]).cumprod() - 1.0
    return result


def compute_sim_metrics(result: pd.DataFrame,
                        l_ms: int = 300_000) -> dict:
    """Compute professional trading metrics from a trade-log DataFrame.

    Args:
        result:  Output of run_trading_simulation.
        l_ms:    Interval length in ms — used to pick annualisation factor.

    Returns:
        Dict with: total_return_pct, sharpe_annualised, max_drawdown_pct,
                   calmar_ratio, win_rate_pct, avg_net_return_bps, n_trades.
    """
    if result is None or len(result) == 0:
        return {"n_trades": 0}

    r = result["net_return"]
    cum = (1.0 + r).cumprod()
    drawdown = 1.0 - cum / cum.cummax()
    max_dd = float(drawdown.max())
    total_ret = float(cum.iloc[-1] - 1.0)
    ann = _ANN.get(l_ms, _ANN[300_000])

    sharpe = float((r.mean() / r.std()) * ann) if r.std() > 0 else 0.0
    calmar = total_ret / max_dd if max_dd > 0 else float("inf")

    return {
        "total_return_pct":   round(total_ret * 100, 4),
        "sharpe_annualised":  round(sharpe, 4),
        "max_drawdown_pct":   round(max_dd * 100, 4),
        "calmar_ratio":       round(calmar, 4),
        "win_rate_pct":       round(float((r > 0).mean() * 100), 4),
        "avg_net_return_bps": round(float(r.mean() * 10_000), 4),
        "n_trades":           int(len(r)),
    }


def run_fee_stress_test(
    model,
    X_test: np.ndarray,
    y_pred_class: np.ndarray,
    intervals: pd.DataFrame,
    t_indices: np.ndarray,
    hold_periods: int = 5,
    l_ms: int = 300_000,
    fee_levels: list = None,
) -> dict:
    """Re-run simulation at multiple fee levels to find the breakeven cost.

    Default fee levels (per order, in decimal):
      0.000003  — paper baseline (Binance VIP7 maker)
      0.0001    — Binance standard maker
      0.00075   — Binance retail taker
      0.001     — Retail taker + 1 bp slippage proxy
    """
    if fee_levels is None:
        fee_levels = [0.000003, 0.0001, 0.00075, 0.001]

    results = {}
    for fee in fee_levels:
        sim = run_trading_simulation(
            model, X_test, y_pred_class, intervals, t_indices,
            fee=fee, hold_periods=hold_periods,
        )
        label = f"fee_{fee:.6f}"
        results[label] = compute_sim_metrics(sim, l_ms)
    return results


# ---------------------------------------------------------------------------
# Quant extension §5.3: confidence-weighted position sizing
# ---------------------------------------------------------------------------

def run_confidence_weighted_simulation(
    model,
    X_test: np.ndarray,
    y_pred_proba: np.ndarray,
    intervals: pd.DataFrame,
    t_indices: np.ndarray,
    fee: float = 0.000003,
    hold_periods: int = 5,
    m: int = 5,
    min_size: float = 0.0,
) -> pd.DataFrame:
    """Long/short backtest with position size proportional to model
    confidence — the implementation-plan §5.3 extension.

    The binary sim treats every prediction as a full-size long or short,
    even when ``P(up)`` is 50.5 %. That throws away the rank-correlation
    signal captured by the Information Coefficient. Here:

        size      = 2·P(up) − 1     ∈ [−1, +1]
        direction = sign(size)
        notional  = |size|

    A 51 % prediction takes a 0.02 position; a 90 % prediction takes a
    0.80 position; a coin-flip prediction (P(up)=0.5) takes 0 size and
    pays 0 fees. Returns scale linearly with size; round-trip fees scale
    with ``|size|``.

    For setups where the IC is meaningful (l=300k, m=24 has IC = 0.035
    in our 11/3-month run) this should:
      • lower turnover and cumulative fees,
      • raise risk-adjusted returns (Sharpe),
      • raise the breakeven fee — the metric that decides whether the
        strategy is commercially viable.

    Args:
        model:        Kept for API parity with run_trading_simulation.
        X_test:       Kept for API parity.
        y_pred_proba: (N, 2) softmax outputs from ``model.predict``.
                      Column 0 is ``P(up)`` per ``make_labels``.
        intervals:    Interval DataFrame with ``vwap_raw`` (real prices).
        t_indices:    (N,) prediction-time row indices into ``intervals``.
        fee:          Fee per order, decimal. Round-trip cost is
                      ``2 × fee × |size|``.
        hold_periods: Bars to hold each position before closing.
        m:            Informational; not used in P&L.
        min_size:     Skip trades with ``|size| < min_size`` entirely
                      (default 0 — keep every trade with non-zero size).

    Returns:
        Trade-log DataFrame in the same schema as
        ``run_trading_simulation``, plus a ``size`` column for diagnostics.
    """
    if "vwap_raw" in intervals.columns:
        vwap = intervals["vwap_raw"].values
    else:
        vwap = intervals["vwap"].values
    timestamps = intervals["interval_end_ms"].values

    if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != 2:
        raise ValueError(
            f"y_pred_proba must be (N, 2), got {y_pred_proba.shape}")
    p_up  = y_pred_proba[:, 0].astype(np.float64)
    sizes = 2.0 * p_up - 1.0   # (-1, +1)

    positions = []
    n = len(t_indices)
    for i in range(n):
        exit_idx = t_indices[i] + hold_periods
        if exit_idx >= len(vwap):
            break
        size = float(sizes[i])
        if abs(size) < min_size:
            continue
        entry_price = vwap[t_indices[i]]
        exit_price  = vwap[exit_idx]
        if entry_price <= 0 or exit_price <= 0:
            continue

        gross_return = (exit_price / entry_price - 1.0) * size
        fee_paid     = 2.0 * fee * abs(size)
        net_return   = gross_return - fee_paid

        positions.append({
            "entry_time":   int(timestamps[t_indices[i]]),
            "exit_time":    int(timestamps[exit_idx]),
            "size":         size,
            "direction":    "long" if size > 0 else "short",
            "entry_price":  float(entry_price),
            "exit_price":   float(exit_price),
            "gross_return": float(gross_return),
            "fee_paid":     float(fee_paid),
            "net_return":   float(net_return),
        })

    if not positions:
        return pd.DataFrame()

    result = pd.DataFrame(positions)
    result["cumulative_return"] = (1.0 + result["net_return"]).cumprod() - 1.0
    return result


def run_fee_stress_test_confidence(
    model,
    X_test: np.ndarray,
    y_pred_proba: np.ndarray,
    intervals: pd.DataFrame,
    t_indices: np.ndarray,
    hold_periods: int = 5,
    l_ms: int = 300_000,
    fee_levels: list = None,
    min_size: float = 0.0,
) -> dict:
    """Fee-stress sweep for the confidence-weighted strategy."""
    if fee_levels is None:
        fee_levels = [0.000003, 0.0001, 0.00075, 0.001]
    results = {}
    for fee in fee_levels:
        sim = run_confidence_weighted_simulation(
            model, X_test, y_pred_proba, intervals, t_indices,
            fee=fee, hold_periods=hold_periods, min_size=min_size,
        )
        results[f"fee_{fee:.6f}"] = compute_sim_metrics(sim, l_ms)
    return results


def find_breakeven_fee(
    model, X_test, y_pred_proba_or_class, intervals, t_indices,
    hold_periods: int = 5,
    l_ms: int = 300_000,
    confidence_weighted: bool = False,
    fee_grid: np.ndarray = None,
) -> float:
    """Bisection over fee levels to find where total_return_pct crosses 0.

    Returns the breakeven fee in decimal. ``np.nan`` if the strategy is
    unprofitable at any positive fee or profitable at any fee in the grid.
    """
    if fee_grid is None:
        fee_grid = np.geomspace(1e-6, 5e-3, 60)
    last_positive = None
    last_negative = None
    for f in fee_grid:
        if confidence_weighted:
            sim = run_confidence_weighted_simulation(
                model, X_test, y_pred_proba_or_class, intervals, t_indices,
                fee=float(f), hold_periods=hold_periods,
            )
        else:
            sim = run_trading_simulation(
                model, X_test, y_pred_proba_or_class, intervals, t_indices,
                fee=float(f), hold_periods=hold_periods,
            )
        m = compute_sim_metrics(sim, l_ms)
        ret = m.get("total_return_pct", -100.0)
        if ret > 0:
            last_positive = float(f)
        else:
            last_negative = float(f)
            break
    if last_positive is None:
        return float("nan")
    return last_positive

