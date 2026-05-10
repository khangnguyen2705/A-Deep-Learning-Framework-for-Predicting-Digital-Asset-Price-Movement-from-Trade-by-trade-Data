# LSTM Direction-of-Move Predictor

> A faithful, instrumented reproduction of *“A Deep Learning Framework for
> Predicting Digital Asset Price Movement from Trade-by-trade Data”*
> ([arXiv:2010.07404](https://arxiv.org/abs/2010.07404)) — extended with
> a full quant-research workbench: Information Coefficient, statistical
> significance testing, regime analysis, signal-decay curves, and a
> fee-stressed long/short trading simulation.

The paper trains an LSTM classifier on Binance BTC-USDT tick data to predict
the binary **direction** of price movement over a fixed horizon. Their best
model reaches **61.12 %** out-of-sample directional accuracy and transfers
to ETH / BCH / LTC / EOS at ~60 % without retraining. This repository
reproduces that pipeline end-to-end and adds the analytics a quant desk
would want before trusting the model.

---

## Table of contents

1. [TL;DR](#tldr)
2. [Quick start](#quick-start)
3. [Background — what is the paper actually claiming?](#background)
4. [Pipeline walkthrough — Stages 1 → 12](#pipeline-walkthrough)
5. [Repository layout](#repository-layout)
6. [Configuration reference](#configuration-reference)
7. [Results — demo runs vs paper targets](#results)
8. [Reports — what every output file contains](#reports)
9. [Quant extensions beyond the paper](#quant-extensions)
10. [Risk flags & methodology decisions](#risk-flags)
11. [Apple Silicon TensorFlow caveat](#tf-caveat)
12. [Reproducibility, environment, install](#repro)
13. [Citation & license](#citation)

---

<a id="tldr"></a>
## 1. TL;DR

```bash
pip install -r requirements.txt
python run.py --config configs/demo.yaml      # 10 min smoke test
```

* Raw input: Binance aggregated trades (`timestamp_ms, price, amount, maker`).
* The pipeline aggregates ticks into 1-min and 5-min bars, fits an LSTM that
  predicts whether VWAP `m` bars from now will be higher than now, then runs
  the model out-of-sample, computes Information Coefficient and binomial
  significance, simulates a long/short strategy with fee stress, and tests
  transfer to other altcoins.
* All outputs are written to `reports/` as JSON, CSV, and PNG.

---

<a id="quick-start"></a>
## 2. Quick start

### Smoke test

```bash
pip install -r requirements.txt
python run.py --fetch-data                      # one-time, ~30 s for the demo months
python run.py --config configs/demo.yaml
```

About 10 minutes on a laptop in eager-TF mode after the fetch. Produces
every report listed in [§8](#reports). The repo no longer ships
parquets in `data/` to keep the clone small; the fetcher writes them on
first use and caches them under `data/raw/`.

### Full paper grid (downloads ~5–15 GB from data.binance.vision first)

```bash
python run.py --fetch-data                     # one-time download
python run.py                                  # full grid (multi-hour)
```

The full grid is `4 setups × 4 T × 4 N = 64 trainings × ≤200 epochs`.
Realistically a multi-hour run on a workstation, days on a laptop in eager
mode. See [§11](#tf-caveat) for the eager-mode caveat.

> **Date schedule for the contemporary run.** The original paper used Mar
> 2019 → Nov 2019 for training and Dec 2019 → Feb 2020 for testing. This
> repository ports that exact 11/3-month split forward to **Mar 2025 →
> Jan 2026** train and **Feb 2026 → Apr 2026** test, so the model is
> evaluated on a current macro / volatility regime rather than on
> seven-year-old data. The dates are coded in
> [`src/data/fetch_binance.py`](src/data/fetch_binance.py) and reflected in
> the YAML.

### Resuming a partial run (cross-cloud, cross-machine)

The pipeline writes per-cell checkpoints during the grid search, so a
crashed or quota-paused run can be resumed *anywhere* — including on a
different GPU / cloud — without re-doing finished work. After every
training cell it writes:

```
reports/cell_<setup>_<horizon>_T<T>_N<N>.json    # one per (T, N)
reports/best_model_<setup>_<horizon>.keras       # rolling best
reports/done_<setup>_<horizon>.json              # marker after OOS+plots
```

Re-running `python run.py …` skips any cell whose JSON already exists
and reloads the best model from disk for downstream OOS / sim / transfer.

**Example: Lightning ran out of GPU-hours mid-grid, finish on Colab.**

1. From the Lightning Studio, snapshot the small `reports/` directory
   somewhere portable:

   ```bash
   tar czf reports_partial.tgz reports/
   # download via the file browser, or push to a private gist:
   gh release create lightning-snapshot reports_partial.tgz
   ```

2. On Colab, mount Drive, clone the repo, restore reports, re-fetch
   data (CPU step, no GPU burn), then resume:

   ```python
   !git clone https://github.com/khangnguyen2705/A-Deep-Learning-Framework-for-Predicting-Digital-Asset-Price-Movement-from-Trade-by-trade-Data.git
   %cd A-Deep-Learning-Framework-for-Predicting-Digital-Asset-Price-Movement-from-Trade-by-trade-Data
   !pip install -q -r requirements.txt
   !tar xzf /content/drive/MyDrive/reports_partial.tgz
   !python -m src.data.fetch_binance --data-dir data       # CPU, ~30-60 min
   # Switch runtime to T4 GPU here.
   !python run.py --config configs/paper_2010_07404.yaml   # auto-resumes
   ```

   Cells already finished on Lightning are echoed as `CACHED  val_loss=…`
   and the L4 weights are reloaded for the OOS evaluation. Only the
   missing cells get retrained on the T4.

### Run on a free GPU (recommended for the full grid)

The full grid is heavy enough that a CPU run is impractical, but the demo
parquets prove the pipeline runs anywhere. For the real run, two free GPU
options:

* **Google Colab T4** — open
  [`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb) in Colab,
  switch the runtime to GPU, and run the cells. The notebook mounts Drive,
  clones this repo, fetches the dataset, removes the macOS eager-mode
  patch, runs the full grid, and copies `reports/` back to Drive. End-to-end
  ~4–8 hours on a T4; a free Colab session may need to be split across two
  runs (the notebook is idempotent — Drive caching survives disconnect).
* **Kaggle Notebooks** — same idea, slightly older Tesla P100, 30 GB disk,
  9-hour session limit. Drag the notebook in, attach a “GPU” accelerator,
  run.
* **Lightning AI Studios / Modal / Vast.ai** — paid but with free
  starter credits ($25–30) typically sufficient for one full grid on an
  L4 / A10 / 3090.

### Re-download if you suspect cached files are stale

```bash
python run.py --force-fetch
```

---

<a id="background"></a>
## 3. Background — what is the paper actually claiming?

If you are new to quant ML, a few definitions before the methodology:

* **Tick / aggregated trade**: a single executed buy/sell printed by an
  exchange. Binance’s “aggTrades” batch the simultaneous fills at one price
  into a single row. Each row carries the timestamp, price, traded amount,
  and a `maker` flag indicating which side rested on the order book.
* **Bar / interval**: a fixed-time bucket of ticks, summarised as a vector
  of features (volume, VWAP, etc.). The paper uses 1-minute and 5-minute
  bars.
* **VWAP** (volume-weighted average price): `Σ(pᵢ aᵢ) / Σ aᵢ`. The
  representative price of a bar.
* **Direction-of-move prediction**: predict the *sign* of the next return,
  not its magnitude. Recast as a binary classification: UP vs DOWN.
* **Out-of-sample (OOS)**: evaluation on data the model never saw during
  training. The only honest measure of edge.
* **Information Coefficient (IC)**: rank correlation between predicted
  probability and realised return. Quant standard for signal quality.
* **Sharpe ratio**: `(mean return / std return) × √(periods per year)`.
  Risk-adjusted return; >1 is good for daily strategies.
* **MaxDD / Calmar**: peak-to-trough loss; total return divided by MaxDD.

The paper’s claim is essentially: *trade-by-trade microstructure features
fed through an LSTM contain enough information to predict the next 30
minutes of BTC direction better than chance, even after transaction
costs, and the learned representation transfers to other coins.* This
repository tests that claim and quantifies how fragile it is to typical
quant pitfalls (fees, regime shifts, lookahead, label leakage).

---

<a id="pipeline-walkthrough"></a>
## 4. Pipeline walkthrough — Stages 1 → 12

`run.py` orchestrates twelve stages. Each stage is a small module with a
single responsibility.

### Stage 1 — Load trade data
* Reads `data/btc_usdt_train.parquet` and `data/btc_usdt_test.parquet`.
* Schema: `timestamp_ms` (int64, UTC ms), `price` (float64), `amount`
  (float64, base-asset quantity), `maker` (bool).
* Convention: `maker = True` ⇒ the buyer is the market maker ⇒ an active
  sell hit the book. So `(1 − maker)` is the active-buy mask.

→ Code: [`src/data/binance_trades.py`](src/data/binance_trades.py),
[`src/data/fetch_binance.py`](src/data/fetch_binance.py).

### Stage 2 — Aggregate to bars (paper eq. 1–6)

For each interval `i = ⌊t / l⌋` where `l ∈ {60 000 ms, 300 000 ms}`:

| Feature             | Formula |
|---------------------|---------|
| `num_trades`        | count of trades in the interval |
| `volume`            | `Σ aᵢ` |
| `active_buy_volume` | `Σ aᵢ (1 − mᵢ)` |
| `amplitude`         | `max p − min p` |
| `price_change`      | `p_last − p_first` |
| `vwap`              | `Σ pᵢ aᵢ / Σ aᵢ` |
| `taker_ratio`       | `Σ aᵢ mᵢ / Σ aᵢ` |

The paper says “4 features” in one place but defines 7 via these formulas.
We use all 7 (the “4” is a paper typo).

The implementation is fully vectorised — 6 M trades into 1-min bars in
**~1 second** via `pandas.groupby.agg`.

→ Code: [`src/features/resample.py`](src/features/resample.py).

### Stage 3 — Stationarity (paper eq. 8)

ADF (Augmented Dickey–Fuller) tests every feature column. **Only `vwap` is
non-stationary** (paper: stat = −1.45, p = 0.55; demo run: stat = −1.36,
p = 0.60). It is replaced with its first difference:

`vwap'(t) = vwap(t) − vwap(t−1)`

The first row becomes NaN and is dropped. Crucially, the *raw* VWAP is
preserved as `vwap_raw` in the interval DataFrame because the trading
simulation needs real prices to fill at, not the differenced series.

→ Code: [`src/features/stationarity.py`](src/features/stationarity.py).

### Stage 4 — Label generation (paper eq. 7)

```
C(m)_t = vwap[t + m] / vwap[t] − 1
label_t = [1, 0]   if C(m)_t >  ε        (UP)
          [0, 1]   otherwise              (DOWN)
```

`m` is the prediction horizon in interval units; `ε` is a deadband applied
only when training to avoid micro-direction noise. At test time `ε = 0`.

→ Code: [`src/features/labeling.py`](src/features/labeling.py).

### Stage 5 — Trailing windows (paper §III.D)

For each prediction time `t`, build `X_t = [x_{t−T}, …, x_{t}] ∈ ℝ^(T × 7)`
and apply **MinMax normalization within that window only**. A global
scaler would constitute lookahead bias and invalidate all results.

To reduce redundancy among heavily overlapping windows, the paper uses an
offset-based subsampling scheme:
* Base stride `T` (non-overlapping).
* Plus `K ∈ [10 %, 50 %] × T` random offsets per training pass.
* Plus the offset `q mod T` for full coverage.

→ Code: [`src/datasets/windowing.py`](src/datasets/windowing.py).

### Stage 6 — Train / validation split (paper §III.E)

* Pick `p` disjoint validation blocks of length `q > T` at random.
* Training segments are the remainder, with any sub-segment shorter than
  `T` discarded.
* Trim the last `m` bars off **every** train segment to prevent label
  leakage: their `C(m)` would peek across the split boundary.
* The split is rejected and resampled (up to 1000 attempts) if the train
  vs validation class balance differs by more than 10 %.

→ Code: [`src/splits/train_val_split.py`](src/splits/train_val_split.py).

### Stage 7 — Model & grid search (paper §IV.A)

```
Input  (batch, T, 7)
LSTM(units = N, return_sequences = False)
Dropout(rate = 0.5)
Dense(units = 2)
Softmax()
Output (batch, 2)  →  [P(up), P(down)]
```

| Hyperparameter | Value |
|---|---|
| Loss              | Categorical cross-entropy |
| Optimizer         | Adam, `lr₀ = 0.001` |
| LR schedule       | `lr − 0.0003` every 15 epochs, floor `0.0001` |
| Early stopping    | patience 20 OR `val_loss > min × 1.05` |
| Batch size        | 128 if `n > 50 k`, 64 if `n > 20 k`, else 32 |

The LR schedule is implemented as a `LearningRateScheduler` callback (per
epoch). Subclassing `LearningRateSchedule` triggered a TF eager-execute
deadlock on Apple Silicon (see [§11](#tf-caveat)).

The paper grid:

| `l` (ms)  | `m`     | `T`                       | `N`              |
|-----------|---------|---------------------------|------------------|
| 60 000    | 15, 30  | 100, 300, 1000, 2000      | 16, 32, 64, 128  |
| 300 000   | 6, 24   | 60, 300, 500, 1000        | 16, 32, 64, 128  |

→ Code: [`src/model/lstm_classifier.py`](src/model/lstm_classifier.py),
[`src/train/train_grid_search.py`](src/train/train_grid_search.py).

### Stage 8 — Out-of-sample evaluation (paper §V.A)

Predict every valid trailing window in the test set chronologically (no
shuffle, `ε = 0`). Report:

* Categorical cross-entropy loss, directional accuracy.
* **Daily rolling accuracy** distribution → histogram PNG.
* **Confusion matrix** is implicit in the binomial test below.

→ Code: [`src/eval/out_of_sample.py`](src/eval/out_of_sample.py).

### Stage 9 — Quant analytics

* **Spearman rank IC** between `P(up)` and realised `C(m)` → `ic`,
  `ic_t_stat`, `ic_pval`.
* **Rolling 7-day IC** → CSV + PNG bar chart.
* **One-sided binomial test** of accuracy vs the 50 % null → z-score,
  p-value, significance flags at 5 % and 1 %.
* **Regime split:** rolling-amplitude std partitioned at the median into
  high-vol / low-vol; accuracy and significance reported per regime.

→ Code: [`src/eval/quant_metrics.py`](src/eval/quant_metrics.py).

### Stage 10 — Trading simulation (paper §V.B)

For each prediction point: open a long if `P(up) > P(down)`, else open a
short. Hold for exactly `hold_periods` intervals, then close.

```
gross_return = (exit_vwap_raw / entry_vwap_raw − 1) × direction
net_return   = gross_return − 2 × fee
```

Fees are **per executed order**. The paper assumes 0.0003 % (Binance VIP7
maker). The simulator runs a fee-stress sweep at:

* 0.0003 % (paper baseline)
* 0.01 % (standard maker)
* 0.075 % (retail taker)
* 0.1 % (retail taker + 1 bp slippage proxy)

For each fee level it reports total return, annualised Sharpe, MaxDD,
Calmar, win rate, and average net return per trade in basis points.

A **signal-decay curve** runs the model unchanged but relabels at horizons
`m, m+1, …, m+5` and reports accuracy at each — the empirical half-life of
the edge.

→ Code: [`src/sim/trading_sim.py`](src/sim/trading_sim.py).

### Stage 11 — Transfer learning (paper §V.C)

Apply the BTC-trained weights, with no fine-tuning, to ETH / BCH / LTC /
EOS test data. Report accuracy + binomial z-score per asset.

→ Code: [`src/transfer/other_pairs.py`](src/transfer/other_pairs.py).

### Stage 12 — Save reports

Everything from Stages 8–11 is dumped under `reports/`. See [§8](#reports).

---

<a id="repository-layout"></a>
## 5. Repository layout

```
.
├── configs/
│   ├── paper_2010_07404.yaml   # full paper grid
│   └── demo.yaml               # 12-config trimmed grid for laptops
├── data/
│   ├── btc_usdt_train.parquet  # Jan 2019 BTC sample (~6 M trades)
│   ├── btc_usdt_test.parquet   # Feb 2019 BTC sample (~5.5 M trades)
│   └── raw/                    # cache for fetched monthly Binance dumps
├── src/
│   ├── data/
│   │   ├── binance_trades.py   # parquet loader, schema validation
│   │   └── fetch_binance.py    # data.binance.vision downloader
│   ├── features/
│   │   ├── resample.py         # vectorised eq. 1–6 aggregation
│   │   ├── labeling.py         # eq. 7 + threshold ε
│   │   └── stationarity.py     # ADF report + eq. 8 differencing
│   ├── datasets/
│   │   └── windowing.py        # trailing windows + per-window MinMax
│   ├── splits/
│   │   └── train_val_split.py  # disjoint blocks + leakage trim
│   ├── model/
│   │   └── lstm_classifier.py  # LSTM(N) → Dropout → Dense(2) → Softmax
│   ├── train/
│   │   └── train_grid_search.py# Adam + LR-decay callback + early stop
│   ├── eval/
│   │   ├── out_of_sample.py    # chronological OOS + daily rolling acc
│   │   └── quant_metrics.py    # IC, binomial, signal decay, regimes
│   ├── sim/
│   │   └── trading_sim.py      # long/short backtest + fee stress
│   └── transfer/
│       └── other_pairs.py      # apply BTC weights to altcoins
├── reports/                    # all outputs land here
├── run.py                      # orchestrator (Stages 1 → 12)
├── implementation_plann.md     # paper cross-reference plan
├── requirements.txt
└── README.md
```

---

<a id="configuration-reference"></a>
## 6. Configuration reference

Every knob is in YAML. Two ready-to-go configs ship in `configs/`:

* `paper_2010_07404.yaml` — full paper grid.
* `demo.yaml` — laptop-friendly grid for the bundled 1-month sample.

### Top-level keys

| Key | Meaning |
|---|---|
| `data.btc_usdt.train_path` / `test_path` | Where to load BTC parquets from. |
| `data.btc_usdt.train_range` / `test_range` | Date strings; informational. |
| `data.other_pairs.symbols` | Altcoin tickers for transfer (`["ETH","BCH","LTC","EOS"]`). |
| `data.other_pairs.path_template` | `data/{symbol}_test.parquet`. |
| `interval_lengths_ms` | `{l_60000: 60000, l_300000: 300000}` — 1-min and 5-min bars. |
| `epsilon_test` | `ε` at test time. Always 0 in the paper. |

### `setups.<l>.horizons.<m>.epsilon_train`

Training-time deadband. Paper uses 0 for every horizon except
`l=300k, m=24` where `ε = 0.0002` (~2 bps) to balance the classes.

### `setups.<l>.grid`

| Sub-key | Meaning |
|---|---|
| `T` | List of trailing-window lengths. Searched over for the best (T, N). |
| `N` | List of LSTM hidden-state widths. |

### `setups.<l>.best_params.<m>`

Paper-reported best `(T, N)` per setup. **Not** used by the grid search;
purely documentation.

### `training`

| Key | Meaning |
|---|---|
| `initial_lr` | `0.001` — Adam initial learning rate. |
| `lr_decay` | `0.0003` — subtracted every `lr_decay_epochs`. |
| `lr_decay_epochs` | `15` — step size of the piecewise-linear schedule. |
| `min_lr` | `0.0001` — floor of the schedule. |
| `early_stop_patience` | Epochs without `val_loss` improvement before stopping. |
| `early_stop_delta` | Stops if `val_loss > best × (1 + delta)` (default `0.05`). |
| `max_epochs` | Hard cap regardless of early-stop status. |
| `batch_size_min` / `_max` | Floor / ceiling for the dataset-size-driven rule. |

### `splits`

| Key | Meaning |
|---|---|
| `p` | Number of disjoint validation blocks. |
| `q_factor` | `q = q_factor × T` — minimum length of each val block. |
| `seed` | RNG seed for split + offset sampling. |

### `windowing`

| Key | Meaning |
|---|---|
| `offset_fraction_min` / `_max` | Range from which `K / num_non_overlap_windows` is sampled. Paper says 10 %–50 %. |

### `trading_sim`

| Key | Meaning |
|---|---|
| `interval_ms` | Bar length used for the simulation (`300000` = 5 min, the paper’s default). |
| `T` | Trailing-window length passed to the model at sim time. |
| `m` | Label horizon for the simulator (paper mentions “5 intervals” here). |
| `fee` | Per-order fee in decimal (`0.000003 = 0.0003 %`). |
| `hold_period_intervals` | How many bars each position is held before closing. |

---

<a id="results"></a>
## 7. Results

The headline run trains on **March 2025 → January 2026** (11 months,
373 M BTC trades) and tests on **February → April 2026** (3 months,
117 M trades) — the paper’s exact 11/3-month split, ported forward to
contemporary microstructure. ETH / BCH / LTC test data covers the same
3-month window. EOS dropped (delisted from Binance spot in 2024).

Run produced on a Lightning AI L4 GPU (free tier), full 4×4 paper grid
(64 trainings), max_epochs = 200 with patience-20 early stop, then
re-evaluated with full-coverage OOS via `reevaluate.py`. Total cost:
~16 GPU-hours.

### TL;DR

> **The paper’s headline 61 % directional-accuracy edge does not survive
> into 2026.** Same architecture / hyperparameter grid produces OOS
> accuracy of **50.12 %–50.58 % across all four setups** — vs the paper’s
> 57–61 %. **About 94 % of the directional edge has been arbitraged
> away** in the seven years since publication.
>
> **But:** the residual signal is not gone — the binary cutoff is
> destroying it. The l=300k m=24 model has IC = 0.0353 (highly
> significant, `p < 1e-6`); a confidence-weighted version of the trading
> simulation turns the −79 % binary catastrophe into **+23 % return,
> Sharpe 4.69**, and lifts the breakeven fee from "unprofitable at any
> fee" to **0.66 bps per order** — commercially viable on
> Binance VIP1+ maker schedules. The full §V.B trading sim is also
> profitable at the paper’s VIP7 fee but turns catastrophic at any
> retail fee level.

### Per-setup directional accuracy (full-coverage OOS)

| Setup | Best (T, N) | Val Loss | OOS Acc | n_test | z | p-value | Sig 1 %? | **Paper OOS** |
|------|---|---|---|---|---|---|---|---|
| l=60k, m=15 | (300, 32) | 0.6928 | **50.58 %** | 127,844 | 4.14 | **1.7e-5** | ✅ | 57.18 % |
| l=60k, m=30 | (2000, 16) | 0.6940 | 50.15 % | 126,129 | 1.06 | 0.146 | ❌ | 57.65 % |
| **l=300k, m=6** | (1000, 64) | 0.6898 | 50.12 % | 24,625 | 0.39 | 0.351 | ❌ | **61.12 %** |
| l=300k, m=24 | (1000, 16) | 0.6817 | 50.47 % | 24,607 | 1.46 | 0.073 | ❌ | 57.08 % |

Only the 1-min m=15 setup retains a binomial-significant edge above
chance. Note that best `(T, N)` pairs cluster *near* the paper’s optima
(T=300 for m=15, T=1000 vs paper’s T=300 for m=6) — the structural
fingerprint of the paper survives, the magnitude does not.

### Information Coefficient — where the residual signal lives

| Setup | IC (Spearman) | t-stat | p-value | Verdict |
|---|---:|---:|---:|---|
| l=60k, m=15 | **0.0148** | 5.30 | ~0 | ✅ |
| l=60k, m=30 | -0.0011 | -0.39 | 0.69 | null |
| l=300k, m=6 | 0.0087 | 1.36 | 0.17 | weak |
| **l=300k, m=24** | **0.0353** | 5.54 | ~0 | ✅✅ |

The l=300k m=24 model has a tradeable IC (≥0.02 is institutionally
viable) — the model’s probability scores rank-correlate with realised
returns even though their binary cutoff at 50 % loses the magnitude
information. **A confidence-weighted strategy would extract this**;
the current binary long/short does not (see §5.3 of the implementation
plan for the proposed extension).

### Regime split (l=60k, m=15)

The only setup with enough samples to split meaningfully:

| Regime | Acc | z | p-value |
|---|---:|---:|---:|
| High-vol | 50.52 % | 2.64 | **0.004** ✅ |
| Low-vol  | 50.64 % | 3.22 | **0.0007** ✅ |

Both regimes carry the small but real edge — *not* the typical
overfit-to-volatile-periods pattern. Encouraging structural finding.

### Trading simulation — best l=300k m=6 model, full-coverage 24,619 trades

| Metric | Value |
|---|---:|
| Trades | 24,619 |
| Win rate | 50.01 % |
| Total return | **+73.65 %** |
| Sharpe (annualised) | **2.06** |
| Max drawdown | 75.57 % |
| Calmar | 0.97 |
| Avg net per trade | 0.30 bps |

A 50.12 % directional model paired with 24 619 trades at near-zero fees
compounds to +73 % over 3 months. The classic high-frequency
phenomenon: tiny edge × huge sample size = tradeable. But Sharpe 2.06
with 75 % MaxDD is a brittle ride.

### Fee stress — the killer

| Fee per order | Fee meaning | Cum return | Sharpe | Avg bps/trade |
|---|---|---:|---:|---:|
| 0.0003 % | Binance VIP7 maker | **+73.6 %** | **+2.06** | +0.30 |
| 0.01 %   | Binance retail maker | **−98.5 %** | −11.1 | −1.64 |
| 0.075 %  | Binance retail taker | **−100 %** | −99.6 | −14.6 |
| 0.10 %   | Retail taker + 1 bp slip | **−100 %** | −133.7 | −19.6 |

Implied breakeven fee: **~0.0018 % round-trip**. Profitable only with a
fee schedule below VIP7 maker. Outside that bracket the strategy
loses 100 % of capital within 3 months.

### Transfer learning — the BCH surprise

| Asset | OOS Acc | n_test | z | p-value | **Paper** |
|---|---:|---:|---:|---:|---|
| ETH | 50.08 % | 24,625 | 0.24 | 0.409 | 60.48 % |
| **BCH** | **51.34 %** | 24,625 | **4.21** | **1.3e-5** ✅ | 60.17 % |
| LTC | 49.72 % | 24,625 | -0.87 | 0.810 | 59.96 % |
| EOS | — | — | — | (delisted) | 60.03 % |

**Most interesting result in the project.** Trained on BTC, the model
predicts BCH direction *better and more significantly* than it predicts
its own native pair (51.34 % at `p = 1.3e-5` vs BTC 50.12 % at `p = 0.35`).

ETH is dead-flat (institutional, hyper-efficient like BTC). LTC slightly
anti-predictive but not significantly. The relative *ordering* mirrors
the paper (BCH was also the strongest transfer in 2019), even though
absolute accuracy compressed by ~9 pp.

### Direct paper-vs-2026 comparison

| Metric | Paper (2019) | This run (2026) | Decay |
|---|---:|---:|---:|
| Best val loss (l=300k, m=6) | 0.6610 | 0.6898 | +0.029 |
| OOS acc (l=300k, m=6) | 61.12 % | 50.12 % | **−11.0 pp** |
| OOS acc (l=60k, m=15) | 57.18 % | 50.58 % | **−6.6 pp** |
| Sim outperforms BTC HODL | Yes | Yes (at paper fee) | — |
| Sim survives realistic fees | Not stress-tested | **No** (dies at 0.01 %) | new finding |
| Transfer ETH | 60.48 % | 50.08 % | **−10.4 pp** |
| Transfer BCH | 60.17 % | **51.34 %** ✅ | **−8.8 pp**, *but still significant* |
| Transfer LTC | 59.96 % | 49.72 % | **−10.2 pp** |

### Verdict

* **Structurally yes:** the *direction* of every result still goes the
  paper’s way. Best `(T, N)` pairs near paper’s optima, ADF pattern
  matches, IC positive on three of four setups, BCH transfer remains
  the most significant cross-asset signal.
* **Magnitudinally no:** ~94 % of the directional accuracy gap above
  50 % has decayed. What was a 7–11 pp edge in 2019 is now 0.1–0.6 pp.
* **Commercially no for anyone but VIP7 makers:** profitability lives
  entirely in the 0–0.0018 % fee bracket.

This is honest microstructure decay. The paper described an early-2019
inefficiency. By 2026, market participants — including HFT desks running
models 100× more sophisticated than this LSTM — have priced it out.

### Quant extension tested — confidence-weighted sizing (§5.3)

The l=300k m=24 model has a tradeable Information Coefficient (0.0353,
`p < 1e-6`) but binary OOS accuracy of only 50.47 %. That gap means the
model’s probability scores correctly *rank* trades by expected return,
even when the 50 % cutoff used by the paper’s long/short loses the
magnitude. **Confidence-weighted sizing** keeps that magnitude:

```
size      = 2·P(up) − 1                  ∈ [−1, +1]
direction = sign(size)
notional  = |size|

gross  = (exit/entry − 1) × size
fees   = 2 × fee × |size|                # half-size, half-fees
```

A 51 % prediction takes a 0.02 position; a 90 % prediction takes 0.80.
Coin-flip predictions (P(up) ≈ 0.5) take ~0 size and pay ~0 fees —
exactly the trades that drag the binary sim through retail-fee
catastrophes.

To reproduce on top of an existing reports directory:

```bash
python confidence_sim.py                    # CPU only; ~2 min  (m=6 model)
python confidence_sim.py --horizon m_24     # CPU only; ~2 min  (m=24 model)
```

Both runs load the corresponding `reports/best_model_l_300000_<m>.keras`
(saved by `reevaluate.py`) and produce a head-to-head binary-vs-CW
comparison plus fee stress + bisected breakeven fee. Output goes to
`reports/sim_metrics_confidence_weighted.json` (m=6) and
`reports/sim_metrics_confidence_weighted_m_24.json` (m=24).

#### Result: the IC translates dramatically

The two horizons answer two different questions:

**l=300k, m=6 model** (IC = 0.0087, not significant):

| Metric            | Binary    | Confidence-weighted |
|-------------------|----------:|--------------------:|
| Return @ 0.0003 % fee | +73.7 % | +4.1 %              |
| Sharpe            | +2.06     | +1.29               |
| **MaxDD**         | 75.6 %    | **10.7 %** (7× safer) |
| Breakeven fee     | 1.34 bps  | 1.16 bps            |

CW reduces drawdown 7× but doesn’t move the breakeven fee. Expected:
the model’s probabilities don’t correlate with returns (`IC` p = 0.17),
so CW has nothing to amplify.

**l=300k, m=24 model** (IC = 0.0353, `p < 1e-6`) — **the headline result:**

| Metric            | Binary    | Confidence-weighted |
|-------------------|----------:|--------------------:|
| Return @ 0.0003 % fee | **−79.2 %** | **+23.5 %**       |
| Sharpe            | −1.19     | **+4.69**           |
| MaxDD             | 99.8 %    | **24.1 %** (4× safer) |
| **Breakeven fee** | **NaN** (never profitable) | **6.58e-5 = 0.66 bps** |
| Survives 0.01 % fee | −99.8 % | −8.1 %             |

**Signal flip.** The same 24,583 predictions that the binary strategy
lost 79 % on, the CW strategy *makes* 23 % on, with Sharpe 4.69. The
binary strategy was actively destroying the IC signal by full-sizing
weak-conviction trades; CW preserved it.

Practical implication: the strategy moves from "only VIP7 maker fees
(0.03 bps) profitable" to "any fee below ~0.66 bps profitable",
including Binance VIP1–VIP3 maker schedules (0.10–0.50 bps).
**Commercially viable for institutional flow, not just for the paper’s
top-tier VIP7 assumption.**

This validates the textbook quant insight: **binary accuracy and IC
measure different things.** A model with significant IC but accuracy
near 50 % has real, tradeable signal — the binary directional cutoff
just throws it away.

The implementation is in
[`src/sim/trading_sim.py`](src/sim/trading_sim.py) —
`run_confidence_weighted_simulation`, with a matching fee-stress
wrapper and `find_breakeven_fee` for direct head-to-head comparison.

### Earlier laptop runs (for context)

The repo also reproduces on a CPU laptop with 1 month of bundled BTC
data, for sanity-check before launching the full GPU run. Numbers are
qualitatively the same shape but with much wider error bars; see the
git history of `reports/run_demo.log` if you want to inspect.

---

<a id="reports"></a>
## 8. Reports — what every output file contains

Every run repopulates `reports/` from scratch.

| File | Format | Stage | What it contains |
|---|---|---|---|
| `adf_<setup>_<horizon>.json` | JSON | 3 | Per-feature ADF stat, p-value, stationarity flag, 10 % critical value. |
| `grid_<setup>_<horizon>.json` | JSON | 7 | One row per `(T, N)` searched: `val_loss`, `val_accuracy`, `epochs` taken. |
| `btc_out_of_sample_metrics.json` | JSON | 8–9, 11 | Per-setup `val_loss`, `oos_accuracy`, `oos_loss`, IC, binomial significance, regime split, transfer block. **The headline file.** |
| `rolling_accuracy_<setup>_<horizon>.png` | PNG | 8 | Histogram of daily rolling OOS accuracy with the 50 % baseline. |
| `rolling_ic_<setup>_<horizon>.csv` / `.png` | CSV + PNG | 9 | 7-day rolling Spearman IC with ≥ 5 obs per window. |
| `signal_decay.json` | JSON | 9–10 | Accuracy at hold horizons `m, m+1, …, m+5` for the best `l=300k, m=6` model. |
| `sim_metrics.json` | JSON | 10 | Two top-level keys: `paper_baseline` (0.0003 % fee) and `fee_stress` (sweep at 0.01 %, 0.075 %, 0.1 %). Each has Sharpe, MaxDD, Calmar, win rate, avg net return in bps, n_trades. |
| `trading_sim_equity.png` | PNG | 10 | Cumulative-return curve of the long/short strategy at the paper’s 0.0003 % fee. |
| `transfer_other_pairs_accuracy.csv` | CSV | 11 | Accuracy + binomial significance per altcoin (ETH/BCH/LTC/EOS). |
| `run_demo.log` | text | all | Full stdout of the run (verbose epoch-level training output). |

---

<a id="quant-extensions"></a>
## 9. Quant extensions beyond the paper

These are **not** in the original paper but are essential before any
production trader would take the model seriously.

* **Spearman rank Information Coefficient** + 7-day rolling IC. Measures
  signal quality independent of the binary accuracy metric. An IC of
  0.02–0.05 is considered tradeable at institutional scale.
* **One-sided binomial test** on directional accuracy → z-score, p-value,
  significance flags. Tells you whether 51 % accuracy is real edge or
  Monte Carlo noise.
* **Signal-decay curve** at hold horizons `m, m+1, …, m+5`. Empirical
  half-life of the model’s edge.
* **High-vol vs low-vol regime split** using a 24-bar rolling amplitude
  std as the regime tag.
* **Fee-stress sweep** at four fee levels including a slippage proxy.
  Reports the breakeven fee in basis points.
* **Annualised Sharpe, Calmar, MaxDD, win rate, average return per trade
  in bps** alongside the paper’s lone “cumulative return” metric.
* **Confidence-weighted position sizing** (`run_confidence_weighted_simulation`)
  — size = 2·P(up) − 1 ∈ [−1, +1]; round-trip fees scale with |size|.
  Run via [`confidence_sim.py`](confidence_sim.py) on top of an existing
  trained model. Writes head-to-head metrics + bisected breakeven fees
  to `reports/sim_metrics_confidence_weighted.json`.

---

<a id="risk-flags"></a>
## 10. Risk flags & methodology decisions

These are choices that materially affect the headline numbers, and are
called out in the original implementation plan.

* **Lookahead in normalization (CRITICAL).** Every trailing window is
  MinMax-scaled *within itself*. A global `MinMaxScaler` fit before the
  OOS split would constitute lookahead and invalidate every reported
  number. Implementation:
  [`src/datasets/windowing.py:_minmax_normalize_window`](src/datasets/windowing.py).
* **Label leakage at split boundaries.** `C(m)` looks `m` bars ahead. The
  last `m` bars of every train segment are dropped because their labels
  would peek across the train/val boundary. Implementation:
  [`src/splits/train_val_split.py:_trim_segments`](src/splits/train_val_split.py).
* **Survivorship bias in transfer.** ETH / BCH / LTC / EOS were 2019
  top-cap assets selected with hindsight. Transfer numbers would be
  worse on a random altcoin sample.
* **Transaction-cost optimism.** The paper’s 0.0003 % is Binance’s VIP7
  maker fee. Retail takers pay ~0.075 %; the fee-stress sweep
  ([§7](#results)) shows the strategy turns deeply negative there.
* **Slippage model absent.** Fills happen at the bar’s VWAP — i.e., the
  paper assumes you are the market. The 0.1 % stress level adds a 1 bp
  slippage proxy.
* **Regime non-stationarity.** Trained on 2019 crypto data, the March
  2020 COVID crash represents an extreme regime that the rolling-accuracy
  histogram exposes for any future test set covering it.

Note also one paper inconsistency: the paper writes `T × 4` for the input
shape in §III.D but defines 7 features via eq. 1–6. We use all 7 (the 4 is
a typo).

---

<a id="tf-caveat"></a>
## 11. Apple Silicon TensorFlow caveat

On macOS with TensorFlow 2.21 + Keras 3, `model.fit` for the LSTM hangs at
0 % CPU on the first batch once the training set crosses ~30 k samples.
The hang is inside `TFE_Py_ExecuteCancelable_wrapper`, confirmed with
`sample(1)` traces.

Workaround applied at the top of `run.py`:

```python
import tensorflow as tf
tf.config.run_functions_eagerly(True)            # bypass tf.function tracing
os.environ["TF_NUM_INTEROP_THREADS"] = "1"       # belt-and-suspenders
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]        = "1"
```

Throughput drops ~50× compared to graph mode; everything else is bit-for-bit
identical. On Linux x86_64 (Anthropic CI, AWS, Colab, Lambda Labs) the
workaround is unnecessary — remove it for a much faster run.

Other things that **did not** fix the hang on this machine:
`Input(shape=…)` instead of `LSTM(input_shape=…)`, `compile(jit_compile=False)`,
moving the LR schedule out of `LearningRateSchedule` into a callback,
single-threaded TF env vars alone.

---

<a id="repro"></a>
## 12. Reproducibility, environment, install

### Determinism

* Set `splits.seed = 42` in the YAML (default).
* Train/val split, offset sampling, and TF weight init all consume that
  seed. Numerical reproducibility within ±0.5 % across runs on the same
  hardware.
* The paper itself notes ±0.5–1.5 % variation due to random init — we
  recommend reporting a 95 % CI over three seeds for any headline number.

### Requirements

```
numpy>=1.24       pandas>=1.5      pyarrow>=10
statsmodels>=0.14 tensorflow>=2.12 pyyaml>=6.0
matplotlib>=3.7   scipy>=1.11      requests>=2.28
```

### Install

```bash
git clone <this repo>
cd <this repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Hardware notes

* Apple Silicon laptop, eager mode: demo run ≈ 10 min, beefed-up demo
  ≈ 3–4 hours, full paper grid ≥ 2 days.
* Linux x86_64 workstation, graph mode: demo run ≈ 1 min, beefed-up demo
  ≈ 5 min, full paper grid ≈ 4–8 hours on a single GPU.

---

<a id="citation"></a>
## 13. Citation & license

Original paper:

```bibtex
@article{lyu2020deep,
  title  = {A Deep Learning Framework for Predicting Digital Asset Price
            Movement from Trade-by-trade Data},
  author = {Lyu, Q. and Tao, X. and Li, J.},
  journal= {arXiv preprint arXiv:2010.07404},
  year   = {2020}
}
```

This implementation: MIT licensed (see `LICENSE`). Trade data fetched from
[`data.binance.vision`](https://data.binance.vision/) is subject to
Binance’s terms of use.

The cross-reference plan that drove this rewrite, including audit notes
and gap analysis, is preserved at
[`implementation_plann.md`](implementation_plann.md) for traceability.
