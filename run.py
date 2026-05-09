"""Main pipeline — arXiv 2010.07404 faithful reproduction + quant extensions.

Run with:
    python run.py                              # full paper pipeline
    python run.py --config configs/demo.yaml  # quick smoke-test
    python run.py --fetch-data                # download data first

Pipeline stages
---------------
1.  Data ingestion  (Binance aggTrades via fetch_binance.py)
2.  Interval resampling  (7 features, l=60k and l=300k ms)
3.  Stationarity fix  (ADF report + vwap differencing)
4.  Label generation  (C(m) + threshold ε)
5.  Trailing-window construction  (per-window min-max norm + offset sampling)
6.  Train/val split  (disconnected temporal blocks)
7.  LSTM grid search  (all (l, m, T, N) combos from the paper)
8.  OOS evaluation  (accuracy, loss, rolling accuracy)
9.  Quant analytics  (IC, significance test, regime breakdown, signal decay)
10. Trading simulation  (long/short, paper fees + stress-test)
11. Transfer learning  (BTC weights → ETH/BCH/LTC/EOS)
12. Report generation  (JSON metrics + PNG plots)
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Workaround for a TF 2.21 + LSTM thread-pool deadlock on Apple Silicon that
# triggers on the first `model.fit` once the training set exceeds a few
# thousand windows. Forcing single-threaded TF execution avoids the hang
# while keeping numerical results identical (just slower per epoch). These
# must be set BEFORE TensorFlow is imported anywhere.
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as _tf  # noqa: E402
# Bypass tf.function graph compilation entirely. The Mac/Keras-3 LSTM hang
# manifests during graph tracing of the first fit step; eager mode trades
# throughput for liveness and lets the pipeline complete.
_tf.config.run_functions_eagerly(True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.binance_trades import (
    load_btc_usdt, load_other_pair,
    btc_train_paths, btc_test_paths, other_pair_paths,
)
from src.data.fetch_binance import fetch_all
from src.datasets.windowing import build_trailing_windows, FEATURE_COLUMNS
from src.eval.out_of_sample import evaluate_out_of_sample, compute_rolling_accuracy
from src.eval.quant_metrics import (
    compute_ic, compute_rolling_ic,
    accuracy_significance, compute_signal_decay,
    tag_regimes, compute_regime_accuracy,
)
from src.features.labeling import compute_Cm, make_labels
from src.features.resample import resample_trades, resample_monthly_paths
from src.features.stationarity import prepare_features
from src.sim.trading_sim import (
    run_trading_simulation, compute_sim_metrics, run_fee_stress_test,
)
from src.splits.train_val_split import create_train_val_split
from src.train.train_grid_search import train_model
from src.transfer.other_pairs import evaluate_transfer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resample_any(trades_or_paths, l_ms: int) -> pd.DataFrame:
    """Resample either an in-memory trades DataFrame or a list of monthly
    parquet paths. Path-based mode streams through one month at a time —
    use it for the full 11-month grid where the concatenated parquet
    would OOM."""
    if isinstance(trades_or_paths, pd.DataFrame):
        return resample_trades(trades_or_paths, l_ms)
    return resample_monthly_paths(list(trades_or_paths), l_ms)


def resample_and_label(trades, l_ms: int, m: int,
                       epsilon: float, adf_report_path: str = None,
                       precomputed_intervals: pd.DataFrame = None):
    """Resample trades → stationarity fix → label.

    Accepts either a trades DataFrame (small datasets / demo) or a list of
    monthly parquet paths (full 11-month grid; streams one month at a
    time). When ``precomputed_intervals`` is supplied the resample step is
    skipped — used to share one resample across multiple horizons within
    the same ``l`` setup.

    IMPORTANT ordering:
      1. Resample to intervals (or reuse the pre-computed bars).
      2. Compute C(m) from the RAW vwap series (before any differencing).
         Paper eq. (7): C(m)_t = price(t+m)/price(t) - 1, where price = vwap.
      3. Apply vwap differencing (drops row 0) — stationary input features.
      4. Drop row 0 from every series to keep alignment.
      5. Filter rows where Cm is NaN (last m bars).

    Returns (intervals_df, stat_features, Cm, labels).
    intervals_df has its FEATURE_COLUMNS replaced with stationary values.
    The 'vwap' column in intervals_df is the DIFFERENCED vwap (for LSTM
    input). The raw vwap used for labelling is computed BEFORE that
    overwrite, and a copy lives in intervals['vwap_raw'] for the trading
    sim.
    """
    if precomputed_intervals is not None:
        intervals = precomputed_intervals.copy()
    else:
        intervals = _resample_any(trades, l_ms)
    raw_features = intervals[FEATURE_COLUMNS].values.astype(np.float64)

    # --- Step 2: label using RAW vwap (before any differencing) ---
    # compute_Cm reads intervals["vwap"] which is still the raw series here
    Cm_full   = compute_Cm(intervals, m, l_ms)   # length = len(intervals)
    labels_full = make_labels(Cm_full, epsilon)

    # --- Step 3: ADF report + vwap differencing (drops row 0) ---
    stat_features = prepare_features(raw_features, report_path=adf_report_path)
    # stat_features.shape[0] == len(intervals) - 1

    # --- Step 4: align everything by dropping row 0 ---
    intervals   = intervals.iloc[1:].reset_index(drop=True)
    Cm_aligned  = Cm_full[1:]
    lbl_aligned = labels_full[1:]
    # Overwrite feature columns with stationary values (vwap now differenced).
    # NOTE: intervals['vwap_raw'] remains untouched so the trading sim and
    # signal-decay analysis can recover real prices.
    intervals[FEATURE_COLUMNS] = stat_features.astype(np.float32)

    # --- Step 5: filter valid (non-NaN Cm) rows ---
    valid = ~np.isnan(Cm_aligned)
    return (
        intervals[valid].reset_index(drop=True),
        stat_features[valid],
        Cm_aligned[valid],
        lbl_aligned[valid],
    )


def build_dataset(segments, intervals, Cm, labels, T, m, of_min, of_max, rng):
    """Build (X, y) tensors from a list of (start, end) segment index pairs."""
    X_list, y_list = [], []
    for seg_start, seg_end in segments:
        seg_iv = intervals.iloc[seg_start:seg_end]
        seg_Cm = Cm[seg_start:seg_end]
        seg_lb = labels[seg_start:seg_end]
        Xs, ys, _ = build_trailing_windows(
            seg_iv, seg_Cm, seg_lb, T, m, of_min, of_max, rng,
        )
        if len(Xs) > 0:
            X_list.append(Xs)
            y_list.append(ys)
    if not X_list:
        return np.array([]), np.array([])
    return np.concatenate(X_list), np.concatenate(y_list)


def build_all_test_windows(
    intervals: pd.DataFrame,
    Cm: np.ndarray,
    labels: np.ndarray,
    T: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ALL valid chronological test windows (no offset sampling).

    Paper Section V.A: 'we will test our model on all the valid input data
    in a chronological order.'
    This generates every prediction point t in [T, N) where Cm[t] is not NaN,
    using per-window min-max normalization (same as training).
    """
    from src.datasets.windowing import _minmax_normalize_window
    feature_data = intervals[FEATURE_COLUMNS].values.astype(np.float32)
    n = len(feature_data)
    X_list, y_list, t_list = [], [], []
    for t in range(T, n):
        if np.isnan(Cm[t]):
            continue
        window = feature_data[t - T:t]
        X_list.append(_minmax_normalize_window(window))
        y_list.append(labels[t])
        t_list.append(t)
    if not X_list:
        return np.array([]), np.array([]), np.array([])
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(t_list, dtype=np.int64),
    )


def save_plot_rolling_accuracy(rolling_acc: pd.DataFrame,
                                path: Path, title: str):
    if len(rolling_acc) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(rolling_acc["accuracy"] * 100, bins=20, edgecolor="black",
            color="#2563EB", alpha=0.85)
    ax.axvline(50, color="red", linestyle="--", linewidth=1, label="50 % baseline")
    ax.set_xlabel("Daily Rolling Accuracy (%)")
    ax.set_ylabel("Frequency (days)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_plot_equity(sim_result: pd.DataFrame, path: Path, title: str):
    if len(sim_result) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    dates = pd.to_datetime(sim_result["exit_time"], unit="ms")
    ax.plot(dates, sim_result["cumulative_return"] * 100,
            linewidth=0.9, color="#16A34A")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.fill_between(dates, 0, sim_result["cumulative_return"] * 100,
                    alpha=0.15, color="#16A34A")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_plot_rolling_ic(rolling_ic_df: pd.DataFrame, path: Path, title: str):
    if len(rolling_ic_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(rolling_ic_df)), rolling_ic_df["ic"], color="#7C3AED", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("7-Day Window")
    ax.set_ylabel("Spearman IC")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str = "configs/paper_2010_07404.yaml",
                 data_dir: str = "data"):
    cfg = load_config(config_path)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg["splits"]["seed"])

    # -----------------------------------------------------------------------
    # Stage 1: Data discovery
    # -----------------------------------------------------------------------
    # Prefer the streaming path (per-month parquets in data/raw/) when
    # available — required for the full 11-month grid which would OOM if
    # loaded as a single concatenated DataFrame. Fall back to the in-memory
    # loader when the user only has the small concatenated parquet (demo).
    print("\n" + "=" * 65)
    print("STAGE 1 — Discovering trade data")
    print("=" * 65)
    train_paths = btc_train_paths(data_dir)
    test_paths  = btc_test_paths(data_dir)
    if train_paths and test_paths:
        print(f"  Streaming mode — {len(train_paths)} train month(s), "
              f"{len(test_paths)} test month(s)")
        train_data = train_paths
        test_data  = test_paths
    else:
        print("  Streaming-mode files not found, falling back to "
              "load_btc_usdt() (in-memory)")
        train_data, test_data = load_btc_usdt(data_dir)
        print(f"  Train trades: {len(train_data):,}")
        print(f"  Test  trades: {len(test_data):,}")

    all_results = {}
    best_model_for_sim = None
    best_intervals_for_sim = None
    best_sim_t_indices = None

    # -----------------------------------------------------------------------
    # Stage 2-9: Per-setup loop (l × m combinations)
    # -----------------------------------------------------------------------
    for setup_name, setup in cfg["setups"].items():
        l = setup["l"]
        print(f"\n{'=' * 65}")
        print(f"SETUP {setup_name}  (l = {l:,} ms = {l//60000} min)")
        print("=" * 65)

        # Resample once per setup and share across all horizons. Saves a
        # full pass over the 11-month dataset for every additional m.
        print(f"  Resampling train data at l={l:,} ms ...", flush=True)
        train_intervals_master = _resample_any(train_data, l)
        print(f"    train intervals: {len(train_intervals_master):,}", flush=True)
        print(f"  Resampling test data at l={l:,} ms ...", flush=True)
        test_intervals_master  = _resample_any(test_data, l)
        print(f"    test  intervals: {len(test_intervals_master):,}", flush=True)

        for horizon_name, horizon in setup["horizons"].items():
            m = horizon["m"]
            eps_train = horizon["epsilon_train"]
            eps_test  = cfg["epsilon_test"]
            print(f"\n  --- Horizon m={m}  (ε_train={eps_train}) ---")

            # Stage 2-3: Stationarity + label generation (resample reused)
            adf_path = str(reports_dir / f"adf_{setup_name}_{horizon_name}.json")
            train_iv, _, train_Cm, train_labels = resample_and_label(
                None, l, m, eps_train,
                adf_report_path=adf_path,
                precomputed_intervals=train_intervals_master,
            )
            test_iv, _, test_Cm, test_labels = resample_and_label(
                None, l, m, eps_test,
                precomputed_intervals=test_intervals_master,
            )
            print(f"  Train intervals (after diff): {len(train_iv):,}")
            print(f"  Test  intervals (after diff): {len(test_iv):,}")

            T_grid  = setup["grid"]["T"]
            N_grid  = setup["grid"]["N"]
            of_min  = cfg["windowing"]["offset_fraction_min"]
            of_max  = cfg["windowing"]["offset_fraction_max"]
            p       = cfg["splits"]["p"]
            q_fac   = cfg["splits"]["q_factor"]
            seed    = cfg["splits"]["seed"]
            n_feats = len(FEATURE_COLUMNS)

            best_val_loss = float("inf")
            best_model    = None
            best_params   = None
            grid_log      = []

            # Stage 7: Grid search
            for T in T_grid:
                val_segs, train_segs, _, _ = create_train_val_split(
                    len(train_iv), T, m, p=p, q_factor=q_fac,
                    seed=seed, Cm=train_Cm, labels=train_labels,
                )
                X_train, y_train = build_dataset(
                    train_segs, train_iv, train_Cm, train_labels,
                    T, m, of_min, of_max, rng)
                X_val, y_val = build_dataset(
                    val_segs, train_iv, train_Cm, train_labels,
                    T, m, of_min, of_max, rng)
                if len(X_train) == 0 or len(X_val) == 0:
                    print(f"    T={T}: skipping (empty train/val)")
                    continue

                for N in N_grid:
                    model, history = train_model(
                        X_train, y_train, X_val, y_val,
                        T, n_feats, N,
                        lr=cfg["training"]["initial_lr"],
                        max_epochs=cfg["training"]["max_epochs"],
                        patience=cfg["training"]["early_stop_patience"],
                        delta=cfg["training"]["early_stop_delta"],
                    )
                    vl  = float(min(history.history["val_loss"]))
                    va  = float(max(history.history["val_accuracy"]))
                    epo = int(len(history.history["loss"]))
                    grid_log.append({"T": T, "N": N, "val_loss": vl,
                                     "val_accuracy": va, "epochs": epo})
                    print(f"    T={T:5d} N={N:3d}  "
                          f"val_loss={vl:.4f}  val_acc={va*100:.2f}%  ep={epo}")
                    if vl < best_val_loss:
                        best_val_loss  = vl
                        best_model     = model
                        best_params    = (T, N)

            if best_model is None:
                print(f"  No model trained for {setup_name}/{horizon_name}, skipping.")
                continue

            best_T, best_N = best_params
            print(f"\n  ✓ Best: T={best_T}, N={best_N}, "
                  f"val_loss={best_val_loss:.4f}")

            # Save grid search log
            grid_path = reports_dir / f"grid_{setup_name}_{horizon_name}.json"
            with open(grid_path, "w") as f:
                json.dump(grid_log, f, indent=2)

            # Stage 8: OOS evaluation (chronological, all valid test windows)
            X_test, y_test, t_idx = build_trailing_windows(
                test_iv, test_Cm, test_labels,
                best_T, m, 1.0, 1.0, rng,   # offset_fraction=1 → all positions
            )
            acc, loss, y_pred_cls, y_true_cls, y_pred_proba = \
                evaluate_out_of_sample(best_model, X_test, y_test,
                                       t_idx, test_iv, epsilon=eps_test)
            print(f"  OOS accuracy: {acc*100:.2f}%   loss: {loss:.4f}")

            # Rolling accuracy plot
            rolling_acc = compute_rolling_accuracy(
                y_true_cls, y_pred_cls, test_iv, t_idx)
            save_plot_rolling_accuracy(
                rolling_acc,
                reports_dir / f"rolling_accuracy_{setup_name}_{horizon_name}.png",
                f"Daily Rolling Accuracy  (l={l}, m={m})",
            )

            # Stage 9a: IC analysis
            ic_result = compute_ic(y_pred_proba[:, 0], test_Cm[t_idx])
            ric = compute_rolling_ic(
                y_pred_proba[:, 0], test_Cm[t_idx],
                test_iv["interval_end_ms"].values[t_idx],
            )
            ric_path = reports_dir / f"rolling_ic_{setup_name}_{horizon_name}.png"
            save_plot_rolling_ic(
                ric, ric_path,
                f"Rolling 7-Day IC  (l={l}, m={m})",
            )
            if len(ric) > 0:
                ric.to_csv(
                    reports_dir / f"rolling_ic_{setup_name}_{horizon_name}.csv",
                    index=False)

            # Stage 9b: Statistical significance
            n_total   = len(y_pred_cls)
            n_correct = int(np.sum(y_pred_cls == y_true_cls))
            sig       = accuracy_significance(n_correct, n_total)

            # Stage 9c: Regime analysis
            test_iv_reg = tag_regimes(test_iv)
            regime_acc  = compute_regime_accuracy(
                y_true_cls, y_pred_cls, test_iv_reg, t_idx)

            key = f"{setup_name}_{horizon_name}"
            all_results[key] = {
                "setup":       setup_name,
                "horizon_m":   m,
                "best_T":      int(best_T),
                "best_N":      int(best_N),
                "val_loss":    round(best_val_loss, 6),
                "oos_accuracy": round(acc, 6),
                "oos_loss":     round(loss, 6),
                "ic":           ic_result,
                "significance": sig,
                "regime_accuracy": regime_acc,
            }

            # Track the best BTC model (l=300k, m=6) for sim + transfer
            if setup_name == "l_300000" and horizon_name == "m_6":
                best_model_for_sim      = best_model
                best_intervals_for_sim  = test_iv
                best_sim_t_indices      = t_idx

    # -----------------------------------------------------------------------
    # Stage 10: Trading simulation (best model: l=300k, m=6, T=300)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("STAGE 10 — Trading simulation")
    print("=" * 65)

    if best_model_for_sim is None:
        print("  No best model available; skipping simulation.")
    else:
        sim_cfg  = cfg["trading_sim"]
        l_sim    = sim_cfg["interval_ms"]
        m_sim    = sim_cfg["m"]
        T_sim    = sim_cfg["T"]
        fee_base = sim_cfg["fee"]
        hold     = sim_cfg["hold_period_intervals"]

        sim_iv, _, sim_Cm, sim_labels = resample_and_label(
            test_data, l_sim, m_sim, cfg["epsilon_test"])
        print(f"  Sim intervals: {len(sim_iv):,}")

        X_sim, y_sim, t_sim = build_trailing_windows(
            sim_iv, sim_Cm, sim_labels, T_sim, m_sim, 1.0, 1.0, rng)
        y_pred_sim = best_model_for_sim.predict(X_sim, verbose=0)
        y_pred_sim_cls = np.argmax(y_pred_sim, axis=1)

        # Paper-baseline simulation
        sim_result = run_trading_simulation(
            best_model_for_sim, X_sim, y_pred_sim_cls, sim_iv, t_sim,
            fee=fee_base, hold_periods=hold, m=m_sim,
        )
        sim_metrics = compute_sim_metrics(sim_result, l_ms=l_sim)
        print(f"  Trades: {sim_metrics.get('n_trades', 0)}")
        print(f"  Total return:    {sim_metrics.get('total_return_pct', 'N/A')} %")
        print(f"  Sharpe (ann):    {sim_metrics.get('sharpe_annualised', 'N/A')}")
        print(f"  Max drawdown:    {sim_metrics.get('max_drawdown_pct', 'N/A')} %")
        print(f"  Win rate:        {sim_metrics.get('win_rate_pct', 'N/A')} %")

        save_plot_equity(
            sim_result,
            reports_dir / "trading_sim_equity.png",
            "Trading Simulation Equity Curve  (l=300k, T=300, fee=0.0003%)",
        )

        # Fee stress test
        fee_stress = run_fee_stress_test(
            best_model_for_sim, X_sim, y_pred_sim_cls, sim_iv, t_sim,
            hold_periods=hold, l_ms=l_sim,
        )
        print("\n  Fee stress test:")
        for label, m_dict in fee_stress.items():
            ret = m_dict.get("total_return_pct", "N/A")
            sh  = m_dict.get("sharpe_annualised", "N/A")
            print(f"    {label}: return={ret}%  sharpe={sh}")

        with open(reports_dir / "sim_metrics.json", "w") as f:
            json.dump({"paper_baseline": sim_metrics, "fee_stress": fee_stress},
                      f, indent=2)

        # Signal decay (uses the sim model). Pass RAW vwap so per-horizon
        # returns are computed from real prices rather than the differenced
        # series that lives in the 'vwap' column post-stationarity.
        decay_vwap = (
            sim_iv["vwap_raw"].values
            if "vwap_raw" in sim_iv.columns
            else sim_iv["vwap"].values
        )
        decay = compute_signal_decay(
            best_model_for_sim,
            sim_iv[FEATURE_COLUMNS].values.astype(np.float32),
            decay_vwap,
            t_sim,
            T=T_sim,
            base_m=m_sim,
            extra_horizons=5,
        )
        with open(reports_dir / "signal_decay.json", "w") as f:
            json.dump(decay, f, indent=2)
        print("\n  Signal decay:")
        for d in decay:
            print(f"    h={d['horizon']}: label_acc={d['label_accuracy']:.4f}  "
                  f"n={d['n_samples']}")

    # -----------------------------------------------------------------------
    # Stage 11: Transfer learning (BTC weights → ETH/BCH/LTC/EOS)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("STAGE 11 — Transfer learning")
    print("=" * 65)

    transfer_results = {}
    if best_model_for_sim is None:
        print("  No best model; skipping transfer.")
    else:
        T_trans = cfg["trading_sim"]["T"]
        for symbol in cfg["data"]["other_pairs"]["symbols"]:
            try:
                # Prefer the streaming path (per-month parquets in
                # data/raw/) when available — same OOM concern as BTC.
                paths = other_pair_paths(data_dir, symbol)
                if paths:
                    other_data = paths
                else:
                    other_data = load_other_pair(data_dir, symbol)
                o_iv, _, o_Cm, o_labels = resample_and_label(
                    other_data, 300_000, 6, cfg["epsilon_test"])
                acc_t, loss_t, yp, yt = evaluate_transfer(
                    best_model_for_sim, o_iv, o_Cm, o_labels, T_trans,
                    epsilon=cfg["epsilon_test"],
                )
                sig_t = accuracy_significance(
                    int(np.sum(yp == yt)), len(yp))
                print(f"  {symbol:5s}: accuracy={acc_t*100:.2f}%  "
                      f"z={sig_t.get('z_score','N/A')}  "
                      f"p={sig_t.get('p_value','N/A')}")
                transfer_results[symbol] = {
                    "accuracy": round(float(acc_t), 6),
                    "loss":     round(float(loss_t), 6),
                    **sig_t,
                }
            except Exception as e:
                print(f"  {symbol}: SKIPPED ({e})")
                transfer_results[symbol] = {"accuracy": None, "error": str(e)}

    all_results["transfer"] = transfer_results

    # -----------------------------------------------------------------------
    # Stage 12: Save all reports
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("STAGE 12 — Saving reports")
    print("=" * 65)

    with open(reports_dir / "btc_out_of_sample_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    tf_df = pd.DataFrame(transfer_results).T
    tf_df.to_csv(reports_dir / "transfer_other_pairs_accuracy.csv")

    print(f"\n  Reports written to {reports_dir}/")
    print("  ✓ btc_out_of_sample_metrics.json")
    print("  ✓ sim_metrics.json")
    print("  ✓ transfer_other_pairs_accuracy.csv")
    print("  ✓ rolling_accuracy_*.png  /  rolling_ic_*.png")

    print(f"\n{'=' * 65}")
    print("PIPELINE COMPLETE")
    print("=" * 65)
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="arXiv 2010.07404 pipeline")
    p.add_argument("--config",     default="configs/paper_2010_07404.yaml")
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--fetch-data", action="store_true",
                   help="Download Binance aggTrades before running the pipeline")
    p.add_argument("--force-fetch", action="store_true",
                   help="Re-download even if files already exist")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.fetch_data or args.force_fetch:
        print("Downloading Binance trade data …")
        fetch_all(args.data_dir, force=args.force_fetch)
    run_pipeline(config_path=args.config, data_dir=args.data_dir)
