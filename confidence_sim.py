"""Re-run the trading simulation with confidence-weighted position sizing.

Loads ``reports/best_model_l_300000_m_6.keras`` (saved by reevaluate.py),
predicts the full 3-month test set, and runs *both* simulations:

  1. Binary long/short (paper §V.B, what's already in
     ``reports/sim_metrics.json``).
  2. Confidence-weighted: size = 2·P(up) − 1, fees scale with |size|.

The two are fee-stressed at the same four fee levels and a bisected
breakeven fee is reported per strategy. Output:

    reports/sim_metrics_confidence_weighted.json

Pure CPU script — no GPU work required, model.predict is the only TF op
and runs fine on a small CPU studio. Total cost: ~2 minutes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# TF env hardening (mirrors run.py / reevaluate.py)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",   "3")

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf

from src.data.binance_trades import btc_test_paths, load_btc_usdt
from src.sim.trading_sim import (
    run_trading_simulation, run_fee_stress_test, compute_sim_metrics,
    run_confidence_weighted_simulation, run_fee_stress_test_confidence,
    find_breakeven_fee,
)
from run import build_all_test_windows, resample_and_label


def _print_summary(name: str, m: dict) -> None:
    print(f"  {name:<25} "
          f"trades={m.get('n_trades', 0):>6}  "
          f"return={m.get('total_return_pct', 0):>+8.2f}%  "
          f"sharpe={m.get('sharpe_annualised', 0):>+7.2f}  "
          f"maxdd={m.get('max_drawdown_pct', 0):>6.2f}%  "
          f"win={m.get('win_rate_pct', 0):>5.2f}%  "
          f"avg_bps={m.get('avg_net_return_bps', 0):>+6.2f}",
          flush=True)


def main(config_path: str = "configs/paper_2010_07404.yaml",
         data_dir: str    = "data",
         horizon: str     = "m_6"):
    cfg = yaml.safe_load(open(config_path))
    reports = Path("reports")
    setup_horizon_key = f"l_300000_{horizon}"

    model_path = reports / f"best_model_{setup_horizon_key}.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run reevaluate.py first to train and "
            f"save the best l=300k {horizon} model.")
    metrics_path = reports / "btc_out_of_sample_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found — expected from run.py / reevaluate.py.")

    metrics  = json.load(open(metrics_path))
    if setup_horizon_key not in metrics:
        raise KeyError(
            f"{setup_horizon_key!r} not found in btc_out_of_sample_metrics.json. "
            f"Available keys: {[k for k in metrics if k != 'transfer']}")
    payload  = metrics[setup_horizon_key]
    best_T   = int(payload["best_T"])
    best_m   = int(payload["horizon_m"])
    ic_val   = payload.get("ic", {}).get("ic", float("nan"))
    print(f"loaded {setup_horizon_key} metadata: best_T={best_T}, "
          f"horizon_m={best_m}, IC={ic_val:.4f}", flush=True)

    print("loading model ...", flush=True)
    model = tf.keras.models.load_model(model_path)

    test_paths = btc_test_paths(data_dir)
    test_data  = test_paths if test_paths else load_btc_usdt(data_dir)[1]

    print("resampling 5-min test bars ...", flush=True)
    sim_iv, _, sim_Cm, sim_labels = resample_and_label(
        test_data, 300_000, best_m, cfg["epsilon_test"])
    print(f"  test intervals (after diff): {len(sim_iv):,}", flush=True)

    print(f"building all valid sim windows at T={best_T} ...", flush=True)
    X_sim, y_sim, t_sim = build_all_test_windows(
        sim_iv, sim_Cm, sim_labels, best_T,
    )
    print(f"  n_windows = {len(X_sim):,}", flush=True)

    print("predicting probabilities ...", flush=True)
    y_pred_proba = model.predict(X_sim, verbose=0)
    y_pred_cls   = np.argmax(y_pred_proba, axis=1)

    fee_base = cfg["trading_sim"]["fee"]

    # -----------------------------------------------------------------
    # Binary baseline (paper §V.B)
    # -----------------------------------------------------------------
    print("\n[1/2] binary long/short ...", flush=True)
    bin_sim     = run_trading_simulation(
        model, X_sim, y_pred_cls, sim_iv, t_sim,
        fee=fee_base, hold_periods=best_m, m=best_m,
    )
    bin_metrics = compute_sim_metrics(bin_sim, l_ms=300_000)
    bin_stress  = run_fee_stress_test(
        model, X_sim, y_pred_cls, sim_iv, t_sim,
        hold_periods=best_m, l_ms=300_000,
    )
    bin_breakeven = find_breakeven_fee(
        model, X_sim, y_pred_cls, sim_iv, t_sim,
        hold_periods=best_m, l_ms=300_000,
        confidence_weighted=False,
    )

    # -----------------------------------------------------------------
    # Confidence-weighted (extension §5.3)
    # -----------------------------------------------------------------
    print("\n[2/2] confidence-weighted (size = 2·P(up) − 1) ...", flush=True)
    cw_sim     = run_confidence_weighted_simulation(
        model, X_sim, y_pred_proba, sim_iv, t_sim,
        fee=fee_base, hold_periods=best_m, m=best_m,
    )
    cw_metrics = compute_sim_metrics(cw_sim, l_ms=300_000)
    cw_stress  = run_fee_stress_test_confidence(
        model, X_sim, y_pred_proba, sim_iv, t_sim,
        hold_periods=best_m, l_ms=300_000,
    )
    cw_breakeven = find_breakeven_fee(
        model, X_sim, y_pred_proba, sim_iv, t_sim,
        hold_periods=best_m, l_ms=300_000,
        confidence_weighted=True,
    )

    # -----------------------------------------------------------------
    # Report + persist
    # -----------------------------------------------------------------
    out = {
        "binary": {
            "paper_baseline":  bin_metrics,
            "fee_stress":      bin_stress,
            "breakeven_fee":   bin_breakeven,
        },
        "confidence_weighted": {
            "paper_baseline":  cw_metrics,
            "fee_stress":      cw_stress,
            "breakeven_fee":   cw_breakeven,
        },
        "context": {
            "best_T":          best_T,
            "horizon_m":       best_m,
            "n_windows":       int(len(X_sim)),
            "ic":              payload.get("ic"),
            "binary_oos_acc":  payload.get("oos_accuracy"),
        },
    }
    # The default m=6 output keeps its historical filename so existing
    # README references continue to work; other horizons get a suffix.
    if horizon == "m_6":
        out_path = reports / "sim_metrics_confidence_weighted.json"
    else:
        out_path = reports / f"sim_metrics_confidence_weighted_{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    # ---- console summary ----
    print("\n" + "=" * 86)
    print("PAPER FEE (0.0003 %) — head-to-head")
    print("=" * 86)
    _print_summary("binary",              bin_metrics)
    _print_summary("confidence-weighted", cw_metrics)

    print("\n" + "=" * 86)
    print("FEE STRESS — total_return_pct")
    print("=" * 86)
    print(f"  {'fee':<20} {'binary':>14} {'confidence-weighted':>22}")
    for k in bin_stress:
        b_ret = bin_stress[k].get("total_return_pct", float("nan"))
        c_ret = cw_stress[k].get("total_return_pct", float("nan"))
        print(f"  {k:<20} {b_ret:>+13.2f}% {c_ret:>+21.2f}%")

    print("\n" + "=" * 86)
    print(f"BREAKEVEN FEE  binary={bin_breakeven:.2e}  "
          f"confidence-weighted={cw_breakeven:.2e}")
    print("=" * 86)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/paper_2010_07404.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--horizon", default="m_6",
        help="Which 5-min horizon to test the CW sim on. The l=300k m=24 "
             "model has a much stronger IC (0.0353 vs 0.0087 for m=6) so it "
             "is a more meaningful test of the IC->trading-edge translation.",
    )
    args = parser.parse_args()
    main(args.config, args.data_dir, horizon=args.horizon)
