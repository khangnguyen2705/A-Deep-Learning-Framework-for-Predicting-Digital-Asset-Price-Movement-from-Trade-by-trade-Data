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

