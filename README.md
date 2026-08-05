# Nonstationarity-Complexity Tradeoff: ATOMS vs Fixed Validation Baselines

A reproducible quantitative-finance experiment suite that investigates how
**model complexity** and **nonstationarity** interact when predicting industry
portfolio returns.  The repository provides data loading, factor construction,
and four benchmark experiments centred on 17 Kenneth French industry portfolios
over the period 1987-09 to 2016-11.

---

## Overview

Empirical asset pricing faces two simultaneous challenges:

| Challenge | Description |
|-----------|-------------|
| **Nonstationarity** | Return distributions shift over time (regime changes, crises, structural breaks). |
| **Complexity** | Richer factor models extract more signal but overfit when distribution shifts occur. |

This repository reproduces experiments from the literature showing that the
**optimal validation window length** trades off these two forces — the core
idea of the **ATOMS** (Adaptive Training-period Optimisation via Model
Selection) framework — and benchmarks it against fixed-window cross-validation
baselines.

---

## Experiments

### Experiment 1 — In-sample vs Out-of-sample Decomposition
Decomposes predictive R² into in-sample fit and out-of-sample generalisation
across rolling windows of varying lengths.  Reveals the bias–variance–
nonstationarity tradeoff for OLS, Ridge, Lasso, and random-forest predictors
applied to the 17 industry portfolios.

### Experiment 2 — Recession-Period Stress Tests
Evaluates predictor panels across three NBER recession windows
(Gulf War 1990, Dot-com bust 2001, Global Financial Crisis 2007–2009)
and the full out-of-sample period 1990–2016.  Compares ATOMS window
selection against fixed 1-, 3-, and 5-year training windows.

### Experiment 3 — Factor Complexity Ablation
Progressively increases the predictor set from 17 (lagged industries only)
→ 20 (+ FF3) → 35 (+ Chen) → 129 (+ Gu characteristics) and measures how
optimal training-window length responds to added complexity.

### Experiment 4 — Walk-forward Tournament
A realistic production simulation using a monthly walk-forward tournament
where each model is re-estimated at the end of every month.  Reports
annualised Sharpe ratios, maximum drawdowns, and turnover for long–short
portfolios constructed from model forecasts.

---

## Repository Structure

```
project/
├── data/
│   ├── __init__.py
│   ├── download_factors.py     # Standalone cache-population script
│   └── cache/                  # Auto-created; Parquet files (git-ignored)
│
├── src/
│   ├── __init__.py
│   └── data_loader.py          # All data loading & factor construction
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data

### Publicly Available Sources

| Dataset | Source | Access |
|---------|--------|--------|
| 17 Industry Portfolios (daily) | Kenneth French Data Library | Free — downloaded via `pandas-datareader` |
| Fama-French 3 Factors (daily) | Kenneth French Data Library | Free — downloaded via `pandas-datareader` |

### Restricted / Unavailable Data — Synthetic Proxies

Two factor sets used in the original papers require either proprietary
databases or are not yet publicly released.  **Synthetic AR(1) proxies**
are used in this repository.

#### Chen et al. (2024) — SDF and Latent Factors
> Chen, Z., et al. (2024). *A Latent Factor Model for the Stochastic Discount
> Factor.* Working paper.

The paper constructs 15 monthly factors (1 SDF, 10 beta-decile, 4 macro
hidden-state) from a proprietary return panel.  These are not publicly
available.

**Proxy**: `generate_chen_factors()` produces synthetic AR(1) series
calibrated to the persistence and volatility parameters described in the
paper.  Monthly values are broadcast to all trading days within each
calendar month.  Results that depend on these factors should be treated as
illustrative rather than a strict replication.

#### Gu, Kelly & Xiu (2020) — Characteristic Long-Short Portfolios
> Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine
> Learning.* Review of Financial Studies, 33(5), 2223–2273.

The 94 characteristic-sorted long-short portfolios require CRSP/WRDS
access, which is not freely available.

**Proxy**: `generate_gu_characteristic_factors()` produces 94 daily
long-short returns via a 5-factor latent model with sparse loadings and
mean-reverting factor dynamics (phi=0.05), matching the approximate
cross-sectional dispersion reported in the paper.

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd project

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Python Version
Python ≥ 3.10 is recommended.

---

## Usage

### 1. Populate the Data Cache

Run the download script once before running any experiment.  It fetches
the Fama-French data from the internet and generates the synthetic factors,
then saves everything as Parquet files under `data/cache/`.

```bash
python data/download_factors.py
```

Expected output:
```
Downloading / generating factor data …
Assembling predictor panel …
Saving to data/cache …
  X: (7365, 131)  |  y: (7365, 17)
  Date range: 1987-09-02 → 2016-11-30
Predictor panel shape: (7365, 131)
Target panel shape:    (7365, 17)
Features: ['lag_Food ', 'lag_Mines', 'lag_Oil  ', 'lag_Clths', 'lag_Durbl'] ...
Date range: 1987-09-02 00:00:00 to 2016-11-30 00:00:00
```

### 2. Load the Panel in Python

```python
from src.data_loader import load_full_panel, get_recession_masks

# Load from cache (fast after first run)
X, y = load_full_panel()

# 129 predictors + 'month' + 'year'  →  131 columns
print(X.shape)   # (N, 131)
print(y.shape)   # (N, 17)

# Boolean masks for stress-test periods
masks = get_recession_masks(X.index)
X_gfc = X.loc[masks['gfc']]
y_gfc = y.loc[masks['gfc']]
print(f"GFC observations: {X_gfc.shape[0]}")
```

### 3. Download Individual Factor Sets

```python
from src.data_loader import (
    download_kf_industries,
    download_ff3_daily,
    generate_chen_factors,
    generate_gu_characteristic_factors,
)

industries = download_kf_industries(start='1990-01-01', end='2020-12-31')
ff3        = download_ff3_daily(start='1990-01-01', end='2020-12-31')
chen       = generate_chen_factors(dates=industries.index)
gu         = generate_gu_characteristic_factors(dates=industries.index)
```

### 4. Build the Predictor Panel Manually

```python
from src.data_loader import build_predictor_panel

X, y = build_predictor_panel(industries, ff3, chen, gu)
# X.shape[1] == 131  (129 features + month + year)
```

---

## Predictor Panel Layout

| Block | Columns | Count | Description |
|-------|---------|-------|-------------|
| Lagged industries | `lag_<industry>` | 17 | Industry *t − 1* returns |
| Fama-French | `Mkt-RF`, `SMB`, `HML` | 3 | Market, size, value factors at *t* |
| Chen factors | `chen_sdf`, `chen_decile_01`…`chen_macro_4` | 15 | SDF + decile + macro |
| Gu chars | `char_ls_01`…`char_ls_94` | 94 | Characteristic L/S portfolios |
| **Total** | | **129** | |
| Grouping | `month`, `year` | 2 | Calendar helpers for CV splits |

---

## Citation

If you use this code, please cite the original papers:

```bibtex
@article{gu2020empirical,
  title   = {Empirical Asset Pricing via Machine Learning},
  author  = {Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal = {Review of Financial Studies},
  volume  = {33},
  number  = {5},
  pages   = {2223--2273},
  year    = {2020}
}

@article{fama1993common,
  title   = {Common Risk Factors in the Returns on Stocks and Bonds},
  author  = {Fama, Eugene F. and French, Kenneth R.},
  journal = {Journal of Financial Economics},
  volume  = {33},
  number  = {1},
  pages   = {3--56},
  year    = {1993}
}
```

---

## Notes on Reproducibility

* **Synthetic factors**: Because the Chen et al. (2024) and Gu et al. (2020)
  factor series are synthetic, quantitative results in Experiments 1–4 will
  differ from the original papers.  The *qualitative* nonstationarity-
  complexity tradeoff pattern is preserved.
* **Random seeds**: All synthetic generators accept a `seed` argument
  (default 42) to ensure full reproducibility.
* **Network access**: Fama-French downloads require an internet connection.
  If the download fails, the loader automatically falls back to synthetic
  data and emits a `warnings.warn` message.
* **Cache**: The `data/cache/` directory is git-ignored.  Delete it and
  re-run `download_factors.py` to force a fresh download.
