# Implementation Plan — arXiv 2010.07404
## *A Deep Learning Framework for Predicting Digital Asset Price Movement from Trade-by-trade Data*
### Perspective: Quantitative Researcher @ Jane Street

---

## 1. Executive Summary

The paper trains an **LSTM classifier on Binance trade-by-trade (tick) data** to predict the
binary direction of BTC-USDT price movement over a fixed future horizon. The best model achieves
**61.1% out-of-sample directional accuracy** on BTC-USDT and transfers to ETH/BCH/LTC/EOS at
~60% without re-training. A naïve long/short simulation outperforms a BTC buy-and-hold.

This plan reproduces the paper faithfully, flags all ambiguities, and extends it with
production-grade quant analytics.

> [!IMPORTANT]
> An existing partial implementation already lives in `/Users/macbookair/Desktop/ahaaa/`.
> This plan maps every module against that codebase and identifies what is missing, incorrect,
> or needs hardening before the pipeline is run.

---

## 2. Paper Methodology — Precise Reconstruction

### 2.1 Raw Data Schema

Each Binance trade is a row vector:
```
d_u = [t(u), p(u), a(u), m(u)]   ∈ ℝ^(1×4)
```

| Field | Symbol | Description |
|-------|--------|-------------|
| `timestamp_ms` | t | UTC milliseconds |
| `price` | p | Execution price in USDT |
| `amount` | a | Base asset quantity (BTC) |
| `maker` | m | `True` = buyer is maker (→ active sell hit the book) |

- **Training range:** ~2019-03-01 to 2019-11-30 (~477k 1-min bars, ~95k 5-min bars)
- **OOS test range:** 2019-12-01 to 2020-03-01

### 2.2 Interval Aggregation

Group index: `i = ⌊t(u) / l⌋`, where `l` ∈ {60 000 ms, 300 000 ms}.

**7 features per interval** (equations 1–6 from paper):

| Feature | Formula |
|---------|---------|
| `num_trades` | count of trades in interval |
| `volume` | Σ aᵢ |
| `active_buy_volume` | Σ aᵢ × (1 − mᵢ) |
| `amplitude` | max(pᵢ) − min(pᵢ) |
| `price_change` | p_last − p_first |
| `vwap` | Σ(pᵢ × aᵢ) / Σ(aᵢ) |
| `taker_ratio` | Σ(aᵢ × mᵢ) / Σ(aᵢ) |

> [!WARNING]
> **Paper inconsistency:** The paper states input shape `T × 4` in one place but defines 7
> features via explicit equations. We use all 7. The `T × 4` reference is a likely typo.

### 2.3 Stationarity (ADF Test)

Run ADF on every feature series. Only `vwap` fails (stat = −1.454, p = 0.55). Apply first
differencing per equation (8):

```
price'(t) = vwap(t) − vwap(t−1)
```

All other 6 features are stationary natively (no transformation needed).

### 2.4 Labeling

```
C(m)_t = price(t+m) / price(t) − 1       [eq. 7]
```

Where `price(t)` = raw VWAP of interval t (undifferenced — labels use ratio of raw prices).

| Setup | m | ε (train) | ε (test) | Target class dist. |
|-------|---|-----------|----------|--------------------|
| l=60k | 15 | 0.000 | 0 | 50.65% / 50.65% |
| l=60k | 30 | 0.000 | 0 | 50.80% / 50.80% |
| l=300k | 6 | 0.000 | 0 | 50.75% / 50.75% |
| l=300k | 24 | 0.0002 | 0 | 51.84% → 50.07% |

Label encoding: `[1, 0]` if `C(m) > ε`, else `[0, 1]`.

### 2.5 Trailing Window Construction

For prediction time `t > T`:
```
X = [x_{t-T}, ..., x_t]   ∈ ℝ^(T × 7)
```
Apply **min-max normalization column-wise within each window** — NOT a global scaler.

**Offset-based redundancy reduction:**
- Base stride = T (non-overlapping)
- For each segment of length q, apply K random offsets ∈ [0, T−1], where K ∈ [10%, 50%] of T
- Always include offset = `q mod T` for full coverage

### 2.6 Train/Validation Split

- Randomly select `p` disconnected validation blocks, each of length `q > T`
- Training = remaining segments; discard if length < T
- Zero intersection between training and validation input windows

### 2.7 Model Architecture

```
Input:  (batch, T, 7)
LSTM(units=N, return_sequences=False)
Dropout(rate=0.5)
Dense(2)
Softmax()
Output: (batch, 2)  →  [P(up), P(down)]
```

| Hyperparameter | Value |
|----------------|-------|
| Loss | Categorical cross-entropy |
| Optimizer | Adam, lr₀=0.001 |
| LR schedule | Decay 0.0003 per 15 epochs, min=0.0001 |
| Early stopping | patience=20 OR val_loss > min × 1.05 |
| Batch size | 128 if n>50k, 64 if n>20k, else 32 |

### 2.8 Hyperparameter Grid

| Interval | m | T grid | N grid |
|----------|---|--------|--------|
| l=60 000 ms | 15, 30 | 100, 300, 1000, 2000 | 16, 32, 64, 128 |
| l=300 000 ms | 6, 24 | 60, 300, 500, 1000 | 16, 32, 64, 128 |

**Optimal results (Table III):**

| Setup | T* | N* | Val Loss | Val Acc |
|-------|----|----|----------|---------|
| l=60k, m=15 | 300 | 16 | 0.6812 | 57.66% |
| l=60k, m=30 | 1000 | 32 | 0.6741 | 58.51% |
| **l=300k, m=6** | **300** | **16** | **0.6610** | **62.09%** |
| l=300k, m=24 | 500 | 16 | 0.6791 | 58.04% |

### 2.9 Out-of-Sample Evaluation

- Test: 2019-12-01 to 2020-03-01, chronological order, ε = 0
- Metrics: overall accuracy, loss, daily rolling accuracy distribution

| Setup | OOS Loss | OOS Accuracy |
|-------|----------|-------------|
| l=60k, m=15 | 0.6845 | 57.18% |
| l=60k, m=30 | 0.6799 | 57.65% |
| **l=300k, m=6** | **0.6720** | **61.12%** |
| l=300k, m=24 | 0.6812 | 57.08% |

### 2.10 Trading Simulation Parameters

| Assumption | Value |
|------------|-------|
| Market impact | None |
| Execution latency | None (fills at last price of interval) |
| Transaction cost | **0.0003% per order** |
| Position isolation | Long and short are separate, do not net |
| Hold period | Exactly m intervals |

**Strategy:** Open long if prediction=UP, open short if prediction=DOWN.

```
gross_return = (exit_price / entry_price − 1) × direction
net_return   = gross_return − fee_entry − fee_exit
```

Best sim model: l=300k, T=300. Paper mentions "next 5 intervals" in the simulation — see §4 gap.

### 2.11 Transfer Learning Results

| Asset | OOS Accuracy |
|-------|-------------|
| ETH | 60.48% |
| BCH | 60.17% |
| LTC | 59.96% |
| EOS | 60.03% |

No fine-tuning; BTC-trained weights applied directly.

---

## 3. Existing Codebase Audit (`/Desktop/ahaaa/`)

| Module | Status | Key Gaps |
|--------|--------|----------|
| `src/data/binance_trades.py` | ✅ Exists | Verify `maker` polarity — paper: `maker=True` = buyer is maker = active sell |
| `src/data/fetch_binance.py` | ✅ Exists | Confirm date ranges cover 2019-03 to 2020-03 |
| `src/features/resample.py` | ✅ Exists | Check all 7 features; handle empty intervals |
| `src/features/labeling.py` | ✅ Complete | Correct — `compute_Cm` and `make_labels` match paper |
| `src/features/stationarity.py` | ❓ Unverified | ADF test + VWAP differencing must be applied before windowing |
| `src/datasets/windowing.py` | ✅ Exists | Verify per-window min-max (not global); offset sampling range |
| `src/splits/train_val_split.py` | ✅ Exists | Verify zero overlap guarantee; label leakage in last m bars |
| `src/model/lstm_classifier.py` | ✅ Complete | Matches paper exactly |
| `src/train/train_grid_search.py` | ✅ Complete | LR schedule, early stopping, batch size all correct |
| `src/eval/out_of_sample.py` | ✅ Exists | Confirm chronological ordering; rolling window = 1 day |
| `src/sim/trading_sim.py` | ✅ Complete | `fee=0.000003` = 0.0003% ✓; add Sharpe/MaxDD metrics |
| `src/transfer/other_pairs.py` | ✅ Exists | Verify no weight updates during evaluation |
| `run.py` | ✅ Complete | Full orchestration in place |
| `configs/paper_2010_07404.yaml` | ✅ Exists | Verify all grid values and epsilon values match §2.4 |

> [!CAUTION]
> **Critical lookahead risk:** Min-max normalization must be computed independently per trailing
> window, NOT fit on the full dataset. A global scaler would constitute a severe lookahead bias
> and invalidate all reported OOS numbers.

> [!CAUTION]
> **Label leakage at segment boundaries:** Because `C(m)_t` looks m bars ahead, the last m
> bars of any training segment should not be used as prediction points (their labels would
> require future data that may fall in a validation window).

---

## 4. Gaps to Fill

### 4.1 Simulation Horizon Ambiguity
The paper trains the best model on `m=6` (5-min intervals = 30-min horizon) but the simulation
section describes holding for "5 intervals." Resolution options:
- **(Recommended)** Run a dedicated `m=5` training pass for the sim, using best `T=300, N=16`
- **(Alternative)** Use `m=6` model and fix hold to 6 bars; document the deviation

### 4.2 Enhanced Simulation Metrics
Current `trading_sim.py` only computes cumulative return. Add:

```python
def compute_sim_metrics(result_df):
    r = result_df["net_return"]
    cum = (1 + r).cumprod()
    dd = 1 - cum / cum.cummax()
    ann_factor = np.sqrt(252 * 24 * 12)   # 5-min bars annualised
    return {
        "total_return_pct":    float((cum.iloc[-1] - 1) * 100),
        "sharpe_annualised":   float(r.mean() / r.std() * ann_factor),
        "max_drawdown_pct":    float(dd.max() * 100),
        "calmar_ratio":        float((cum.iloc[-1]-1) / dd.max()),
        "win_rate_pct":        float((r > 0).mean() * 100),
        "avg_net_return_bps":  float(r.mean() * 10000),
        "n_trades":            len(r),
    }
```

### 4.3 Information Coefficient (IC) Analysis
IC = Spearman rank correlation between `P(up)` and realised `C(m)`. Measures signal quality
independently of the binary accuracy metric:

```python
from scipy.stats import spearmanr

def compute_ic(prob_up, realized_cm):
    ic, pval = spearmanr(prob_up, realized_cm)
    return {"ic": round(ic, 4), "ic_t_stat": ic / np.sqrt((1-ic**2)/(len(prob_up)-2)),
            "ic_pval": pval}

def rolling_ic(prob_up, realized_cm, timestamps, window="7D"):
    df = pd.DataFrame({"p": prob_up, "r": realized_cm},
                      index=pd.to_datetime(timestamps, unit="ms"))
    return df.groupby(pd.Grouper(freq=window)).apply(
        lambda g: spearmanr(g["p"], g["r"])[0] if len(g) > 5 else np.nan)
```

### 4.4 Statistical Significance of Accuracy
61% on N samples requires a binomial test before claiming "real edge":

```python
from scipy.stats import binomtest

def accuracy_significance(n_correct, n_total, null_p=0.5):
    result = binomtest(n_correct, n_total, null_p, alternative="greater")
    z = (n_correct/n_total - null_p) / np.sqrt(null_p*(1-null_p)/n_total)
    return {"accuracy": n_correct/n_total, "z_score": round(z, 3),
            "p_value": result.pvalue, "significant_5pct": result.pvalue < 0.05}
```

### 4.5 Regime Analysis
Paper notes accuracy degrades in low-volatility regimes. Add tagging and split reporting:

```python
def tag_regimes(intervals, vol_col="amplitude", window=24):
    rv = intervals[vol_col].rolling(window).std()
    intervals["regime"] = np.where(rv > rv.median(), "high_vol", "low_vol")
    return intervals
# Report accuracy, Sharpe, win_rate by regime → reports/regime_breakdown.json
```

### 4.6 Signal Decay Analysis
For each trained model, evaluate accuracy at hold horizons m, m+1, …, m+5 to measure
edge half-life. Expect decay as market incorporates the information.

### 4.7 Realistic Transaction Cost Stress-Test
0.0003% = Binance VIP7 maker fee. Retail takers pay ~0.075%. Re-run sim at:
- 0.0003% (paper baseline)
- 0.01% (realistic maker)
- 0.075% (retail taker)
- 0.1% (with slippage)

Report breakeven fee in basis points as a key output metric.

---

## 5. Beyond the Paper — Architecture Enhancements

### 5.1 LSTM + Self-Attention
```python
# After LSTM(return_sequences=True), add MultiHeadAttention
# Expected gain: ~0.5–1% accuracy from temporal importance weighting
```

### 5.2 Ensemble Over Seeds
Train best `(l, m, T, N)` with 5 random seeds → average softmax probabilities.
Expected accuracy gain: ~0.5% from variance reduction.

### 5.3 Confidence-Weighted Position Sizing
```python
# Size = 2 * P(correct direction) - 1  ∈ [-1, +1]
# Net return = gross * size - fee * 2 * |size|
# Expected Sharpe improvement vs binary signal
```

### 5.4 Multi-Task Learning
Shared LSTM backbone, separate heads for m=6 and m=24.
Shared representation may improve generalisation across volatility regimes.

---

## 6. Verification Targets

Reproduce within ±1% absolute accuracy and ±0.01 loss:

| Checkpoint | Target |
|------------|--------|
| Val accuracy (l=300k, m=6) | **62.09%** |
| Val loss (l=300k, m=6) | **0.6610** |
| OOS accuracy BTC (l=300k, m=6) | **61.12%** |
| OOS accuracy ETH (transfer) | **60.48%** |
| OOS accuracy BCH (transfer) | **60.17%** |
| OOS accuracy LTC (transfer) | **59.96%** |
| OOS accuracy EOS (transfer) | **60.03%** |
| Sim outperforms BTC buy-and-hold | Yes (qualitative) |
| Sim has significant low-vol drawdowns | Yes (qualitative) |

> [!NOTE]
> Exact replication may shift ±0.5–1.5% due to random initialisation, unspecified `p`/`q`
> split parameters, and offset sampling randomness. Run with `seed=42` and report 95% CI
> over 3 seeds.

---

## 7. Open Questions

| # | Question | Impact |
|---|----------|--------|
| 1 | Paper says `T × 4` input but defines 7 features — use 7? | Medium |
| 2 | Exact `p` and `q` for train/val split (not in paper) | High |
| 3 | Sim holds for 5 bars but best model is m=6 — same model? | Medium |
| 4 | VWAP differencing: applies to input feature only, not label price? | High |
| 5 | `maker=True` = buyer is maker = active **sell**. Confirm `active_buy_volume = Σ aᵢ(1−mᵢ)` ✓ | High |
| 6 | Batch size: constant per dataset size, or decays during training? | Low |

---

## 8. Risk Flags

> [!CAUTION]
> **Lookahead in normalization:** Global `MinMaxScaler` fit before OOS split invalidates all
> results. Must normalize within each individual trailing window.

> [!CAUTION]
> **Label leakage at split boundaries:** Last m bars of each training segment must be excluded
> from input generation since their labels look into the validation period.

> [!WARNING]
> **Survivorship bias in transfer:** ETH/BCH/LTC/EOS are all 2019 top-cap assets — selected
> with hindsight. Results would be worse on a random sample of 2019 altcoins.

> [!WARNING]
> **Transaction cost underestimation:** 0.0003% is VIP7 maker fee. Stress-test at retail taker
> rate (0.075%) before claiming the strategy is commercially viable.

> [!WARNING]
> **No slippage model:** Fills at last price assumes perfect execution. Add 0.5 × spread
> (amplitude / vwap) as a slippage proxy for realistic performance estimates.

> [!NOTE]
> **Regime stationarity:** Model trained on 2019 crypto data. March 2020 COVID crash represents
> an extreme regime — accuracy may degrade significantly. Report rolling accuracy during that
> sub-period explicitly.
