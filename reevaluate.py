"""Re-evaluate an existing run on the FULL test set.

Why: the original ``run.py`` used ``build_trailing_windows`` with offset
sampling for the OOS step. At large T (1000+) on a 3-month 5-min test
set, that scheme only covered ~660 of ~25 000 valid prediction points,
making the headline ``l=300k, m=6`` accuracy a 660-sample noise reading.
Paper §V.A specifies "test our model on all the valid input data in a
chronological order" — i.e., enumerate every valid t.

What this script does, given an existing ``reports/`` directory:

  1. Reads ``grid_<setup>_<horizon>.json`` to pick the best (T, N) per
     horizon (lowest val_loss).
  2. Re-trains that single (T, N) with the same seed and dataset
     construction as the main pipeline (so the train/val split is bit-
     identical to the original — only the test-time windowing changes).
  3. Runs OOS via ``build_all_test_windows`` for honest 25 k-sample
     numbers on every horizon.
  4. Re-runs the trading simulation, fee-stress sweep, signal-decay,
     and ETH/BCH/LTC transfer at full coverage, using the freshly
     trained best model for ``l=300k, m=6``.
  5. Overwrites ``btc_out_of_sample_metrics.json``, ``sim_metrics.json``,
     ``signal_decay.json``, ``transfer_other_pairs_accuracy.csv`` and
     the rolling-accuracy / IC plots in place.

Total cost on Lightning's L4 with the cached resampled data: ~1.5–2
GPU-hours for 4 retrainings + 4 full-coverage predictions.

Run:

    python reevaluate.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# TF env hardening (mirrors run.py)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",   "3")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import yaml

from src.data.binance_trades import (
    btc_train_paths, btc_test_paths, other_pair_paths, load_other_pair,
)
from src.datasets.windowing import build_trailing_windows, FEATURE_COLUMNS
from src.eval.out_of_sample import (
    evaluate_out_of_sample, compute_rolling_accuracy,
)
from src.eval.quant_metrics import (
    compute_ic, compute_rolling_ic, accuracy_significance,
    compute_signal_decay, tag_regimes, compute_regime_accuracy,
)
from src.features.labeling import compute_Cm, make_labels
from src.features.resample import resample_trades, resample_monthly_paths
from src.features.stationarity import prepare_features
from src.sim.trading_sim import (
    run_trading_simulation, compute_sim_metrics, run_fee_stress_test,
)
from src.splits.train_val_split import create_train_val_split
from src.train.train_grid_search import train_model

# Re-use helpers from the main pipeline. ``run`` triggers the TF env
# defaults that file sets, plus its eager-mode patch (no-op on Linux GPU).
from run import (
    _resample_any, build_all_test_windows, build_dataset,
    resample_and_label, save_plot_rolling_accuracy,
    save_plot_equity, save_plot_rolling_ic,
)


# ---------------------------------------------------------------------------
# Per-horizon re-evaluation
# ---------------------------------------------------------------------------

def _read_best_TN(grid_path: Path) -> tuple[int, int, float]:
    """Pick the (T, N) cell with the lowest val_loss from a grid log."""
    with open(grid_path) as f:
        grid = json.load(f)
    best = min(grid, key=lambda c: c["val_loss"])
    return int(best["T"]), int(best["N"]), float(best["val_loss"])


def reevaluate_horizon(
    setup_name: str, horizon_name: str, setup_l: int, horizon_m: int,
    eps_train: float, eps_test: float,
    train_master: pd.DataFrame, test_master: pd.DataFrame,
    cfg: dict, reports_dir: Path,
    rng: np.random.Generator,
) -> tuple[dict, "tf.keras.Model", pd.DataFrame, np.ndarray]:
    grid_path = reports_dir / f"grid_{setup_name}_{horizon_name}.json"
    best_T, best_N, prior_val_loss = _read_best_TN(grid_path)
    print(f"\n  --- {setup_name}/{horizon_name} : best (T={best_T}, N={best_N})"
          f"  prior val_loss={prior_val_loss:.4f} ---", flush=True)

    train_iv, _, train_Cm, train_labels = resample_and_label(
        None, setup_l, horizon_m, eps_train,
        precomputed_intervals=train_master,
    )
    test_iv, _, test_Cm, test_labels = resample_and_label(
        None, setup_l, horizon_m, eps_test,
        precomputed_intervals=test_master,
    )
    print(f"    train intervals (after diff): {len(train_iv):,}", flush=True)
    print(f"    test  intervals (after diff): {len(test_iv):,}", flush=True)

    # Same split + offset RNG seed as the original grid run, so the train
    # tensor is bit-identical to the original best-cell training pass.
    val_segs, train_segs, _, _ = create_train_val_split(
        len(train_iv), best_T, horizon_m,
        p=cfg["splits"]["p"],
        q_factor=cfg["splits"]["q_factor"],
        seed=cfg["splits"]["seed"],
        Cm=train_Cm, labels=train_labels,
    )
    of_min = cfg["windowing"]["offset_fraction_min"]
    of_max = cfg["windowing"]["offset_fraction_max"]

    X_train, y_train = build_dataset(
        train_segs, train_iv, train_Cm, train_labels,
        best_T, horizon_m, of_min, of_max, rng)
    X_val, y_val = build_dataset(
        val_segs, train_iv, train_Cm, train_labels,
        best_T, horizon_m, of_min, of_max, rng)
    print(f"    retraining on {len(X_train):,} train / {len(X_val):,} val "
          f"windows", flush=True)

    model, history = train_model(
        X_train, y_train, X_val, y_val,
        best_T, len(FEATURE_COLUMNS), best_N,
        lr=cfg["training"]["initial_lr"],
        max_epochs=cfg["training"]["max_epochs"],
        patience=cfg["training"]["early_stop_patience"],
        delta=cfg["training"]["early_stop_delta"],
    )
    val_loss = float(min(history.history["val_loss"]))
    val_acc  = float(max(history.history["val_accuracy"]))
    epochs   = int(len(history.history["loss"]))
    print(f"    retrained val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%"
          f"  ep={epochs}", flush=True)

    model_path = reports_dir / f"best_model_{setup_name}_{horizon_name}.keras"
    model.save(model_path)

    # FULL-coverage OOS — the whole point of this script.
    print(f"    building all valid test windows at T={best_T} ...", flush=True)
    X_test, y_test, t_idx = build_all_test_windows(
        test_iv, test_Cm, test_labels, best_T,
    )
    print(f"    full-coverage n_test = {len(X_test):,}", flush=True)

    acc, loss, y_pred_cls, y_true_cls, y_pred_proba = evaluate_out_of_sample(
        model, X_test, y_test, t_idx, test_iv, epsilon=eps_test,
    )
    print(f"    OOS acc={acc*100:.2f}%  loss={loss:.4f}", flush=True)

    # Plots + IC
    rolling_acc = compute_rolling_accuracy(
        y_true_cls, y_pred_cls, test_iv, t_idx)
    save_plot_rolling_accuracy(
        rolling_acc,
        reports_dir / f"rolling_accuracy_{setup_name}_{horizon_name}.png",
        f"Daily Rolling Accuracy  (l={setup_l}, m={horizon_m})",
    )
    ic_result = compute_ic(y_pred_proba[:, 0], test_Cm[t_idx])
    ric = compute_rolling_ic(
        y_pred_proba[:, 0], test_Cm[t_idx],
        test_iv["interval_end_ms"].values[t_idx])
    save_plot_rolling_ic(
        ric,
        reports_dir / f"rolling_ic_{setup_name}_{horizon_name}.png",
        f"Rolling 7-Day IC  (l={setup_l}, m={horizon_m})",
    )
    if len(ric) > 0:
        ric.to_csv(
            reports_dir / f"rolling_ic_{setup_name}_{horizon_name}.csv",
            index=False,
        )

    sig = accuracy_significance(
        int(np.sum(y_pred_cls == y_true_cls)), len(y_pred_cls))
    test_iv_reg = tag_regimes(test_iv)
    regime_acc  = compute_regime_accuracy(
        y_true_cls, y_pred_cls, test_iv_reg, t_idx)

    payload = {
        "setup":          setup_name,
        "horizon_m":      horizon_m,
        "best_T":         best_T,
        "best_N":         best_N,
        "val_loss":       round(val_loss, 6),
        "oos_accuracy":   round(acc, 6),
        "oos_loss":       round(loss, 6),
        "ic":             ic_result,
        "significance":   sig,
        "regime_accuracy": regime_acc,
    }
    return payload, model, test_iv, t_idx


# ---------------------------------------------------------------------------
# Trading sim + transfer at full coverage
# ---------------------------------------------------------------------------

def reevaluate_sim_transfer(
    cfg: dict, data_dir: str, reports_dir: Path,
    btc_test_data, best_model, best_T: int, best_m: int,
    all_results: dict,
):
    print(f"\nSTAGE 10 — Trading simulation (T={best_T}, m={best_m}, "
          f"full coverage)", flush=True)
    sim_iv, _, sim_Cm, sim_labels = resample_and_label(
        btc_test_data, 300_000, best_m, cfg["epsilon_test"])
    X_sim, y_sim, t_sim = build_all_test_windows(
        sim_iv, sim_Cm, sim_labels, best_T,
    )
    print(f"  sim windows: {len(X_sim):,}", flush=True)

    y_pred_sim     = best_model.predict(X_sim, verbose=0)
    y_pred_sim_cls = np.argmax(y_pred_sim, axis=1)

    fee_base = cfg["trading_sim"]["fee"]
    sim_result = run_trading_simulation(
        best_model, X_sim, y_pred_sim_cls, sim_iv, t_sim,
        fee=fee_base, hold_periods=best_m, m=best_m,
    )
    sim_metrics = compute_sim_metrics(sim_result, l_ms=300_000)
    print(f"  trades={sim_metrics.get('n_trades', 0)}  "
          f"return={sim_metrics.get('total_return_pct', 0)}%  "
          f"sharpe={sim_metrics.get('sharpe_annualised', 0)}", flush=True)

    save_plot_equity(
        sim_result, reports_dir / "trading_sim_equity.png",
        f"Trading Simulation Equity Curve  "
        f"(l=300k, T={best_T}, m={best_m}, fee=0.0003%)",
    )

    fee_stress = run_fee_stress_test(
        best_model, X_sim, y_pred_sim_cls, sim_iv, t_sim,
        hold_periods=best_m, l_ms=300_000,
    )
    with open(reports_dir / "sim_metrics.json", "w") as f:
        json.dump({"paper_baseline": sim_metrics, "fee_stress": fee_stress},
                  f, indent=2)

    # Signal decay
    decay_vwap = (
        sim_iv["vwap_raw"].values
        if "vwap_raw" in sim_iv.columns
        else sim_iv["vwap"].values
    )
    decay = compute_signal_decay(
        best_model,
        sim_iv[FEATURE_COLUMNS].values.astype(np.float32),
        decay_vwap,
        t_sim, T=best_T, base_m=best_m, extra_horizons=5,
    )
    with open(reports_dir / "signal_decay.json", "w") as f:
        json.dump(decay, f, indent=2)

    # Transfer
    print("\nSTAGE 11 — Transfer learning (full coverage)", flush=True)
    transfer_results: dict = {}
    for symbol in cfg["data"]["other_pairs"]["symbols"]:
        try:
            paths = other_pair_paths(data_dir, symbol)
            other_data = paths if paths else load_other_pair(data_dir, symbol)
            o_iv, _, o_Cm, o_labels = resample_and_label(
                other_data, 300_000, best_m, cfg["epsilon_test"])
            X_o, y_o, _ = build_all_test_windows(o_iv, o_Cm, o_labels, best_T)
            if len(X_o) == 0:
                raise ValueError("no valid test windows")
            y_pred_o     = best_model.predict(X_o, verbose=0)
            y_pred_o_cls = np.argmax(y_pred_o, axis=1)
            y_true_o_cls = np.argmax(y_o, axis=1)
            acc_t  = float(np.mean(y_pred_o_cls == y_true_o_cls))
            loss_t = float(best_model.evaluate(X_o, y_o, verbose=0)[0])
            sig_t  = accuracy_significance(
                int(np.sum(y_pred_o_cls == y_true_o_cls)), len(y_pred_o_cls))
            print(f"  {symbol:5s}: acc={acc_t*100:.2f}%  "
                  f"n={len(y_pred_o_cls):,}  p={sig_t.get('p_value', 'N/A')}",
                  flush=True)
            transfer_results[symbol] = {
                "accuracy": round(acc_t, 6),
                "loss":     round(loss_t, 6),
                **sig_t,
            }
        except Exception as e:
            print(f"  {symbol}: SKIPPED ({e})", flush=True)
            transfer_results[symbol] = {"accuracy": None, "error": str(e)}

    all_results["transfer"] = transfer_results

    pd.DataFrame(transfer_results).T.to_csv(
        reports_dir / "transfer_other_pairs_accuracy.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "configs/paper_2010_07404.yaml",
         data_dir: str    = "data"):
    cfg = yaml.safe_load(open(config_path))
    reports_dir = Path("reports")
    rng = np.random.default_rng(cfg["splits"]["seed"])

    print("=" * 65)
    print("RE-EVALUATION (full-coverage OOS) — patches the underpowered "
          "OOS samples in the original run.")
    print("=" * 65)

    # Stage 1: discover data (streaming if possible).
    train_paths = btc_train_paths(data_dir)
    test_paths  = btc_test_paths(data_dir)
    if train_paths and test_paths:
        train_data = train_paths
        test_data  = test_paths
        print(f"  Streaming mode — {len(train_paths)} train month(s), "
              f"{len(test_paths)} test month(s)", flush=True)
    else:
        from src.data.binance_trades import load_btc_usdt
        train_data, test_data = load_btc_usdt(data_dir)

    all_results: dict = {}
    best_model_for_sim = None
    best_T_for_sim     = None
    best_m_for_sim     = None

    for setup_name, setup in cfg["setups"].items():
        l = setup["l"]
        print(f"\n{'=' * 65}")
        print(f"SETUP {setup_name}  (l={l:,} ms)")
        print("=" * 65)
        print(f"  resampling train at l={l:,} ms ...", flush=True)
        train_master = _resample_any(train_data, l)
        print(f"    train intervals: {len(train_master):,}", flush=True)
        print(f"  resampling test at l={l:,} ms ...", flush=True)
        test_master = _resample_any(test_data, l)
        print(f"    test  intervals: {len(test_master):,}", flush=True)

        for horizon_name, horizon in setup["horizons"].items():
            payload, model, test_iv, t_idx = reevaluate_horizon(
                setup_name, horizon_name, l,
                horizon["m"],
                horizon["epsilon_train"], cfg["epsilon_test"],
                train_master, test_master,
                cfg, reports_dir, rng,
            )
            key = f"{setup_name}_{horizon_name}"
            all_results[key] = payload

            with open(reports_dir / f"done_{setup_name}_{horizon_name}.json",
                      "w") as f:
                json.dump({"key": key, "payload": payload}, f, indent=2)

            if setup_name == "l_300000" and horizon_name == "m_6":
                best_model_for_sim = model
                best_T_for_sim     = payload["best_T"]
                best_m_for_sim     = payload["horizon_m"]

    # Sim + transfer using the freshly-trained l=300k m=6 model.
    if best_model_for_sim is not None:
        reevaluate_sim_transfer(
            cfg, data_dir, reports_dir,
            test_data, best_model_for_sim,
            best_T_for_sim, best_m_for_sim,
            all_results,
        )

    with open(reports_dir / "btc_out_of_sample_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 65}")
    print("RE-EVALUATION COMPLETE — reports/ now reflects full-coverage OOS")
    print("=" * 65)
    print("  ✓ btc_out_of_sample_metrics.json")
    print("  ✓ sim_metrics.json")
    print("  ✓ signal_decay.json")
    print("  ✓ transfer_other_pairs_accuracy.csv")
    print("  ✓ best_model_*.keras (4 saved models)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/paper_2010_07404.yaml")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    main(args.config, args.data_dir)
