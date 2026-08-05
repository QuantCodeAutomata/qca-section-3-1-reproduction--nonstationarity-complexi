# Repository Knowledge

## Project: Quantitative Finance Experiment — ATOMS Model Selection

### Structure
```
src/
  evaluation.py   — OOS R² metrics (zero-benchmark convention)
  models.py       — CandidateModel dataclass + 168-candidate catalogue + sklearn wrappers
  atoms.py        — ATOMS algorithm (adaptive tournament model selection)
  trading.py      — Sign-based trading signals and wealth accumulation
data/             — (data directory, currently empty)
```

### Key Design Decisions

#### OOS R² (evaluation.py)
- Uses paper's **zero-benchmark** convention: `R² = 1 - SSE / sum(y²)`
- Denominator is sum of **squared actuals**, NOT variance (demeaned)
- Can be negative when the model is worse than a zero forecast

#### 168 Candidates (models.py)
- 28 model/hyperparameter combos × 6 window lengths = 168 total
  - Ridge: 5 alphas `[1e-3, 10^-1.5, 1, 10^1.5, 1e3]`
  - Lasso: 5 alphas `[1e-5, 10^-3.5, 1e-2, 10^-0.5, 10]`
  - ElasticNet: 3 alpha × 3 l1_ratio = 9
  - RF: 3 n_estimators × 3 max_depth = 9
  - Windows: `[1, 4, 16, 64, 256, None]` (None = all history)
- `model_id` format: `{ModelClass}_alpha{:.4g}_w{window|'all'}`
- sklearn intercept is **not** penalised under `fit_intercept=True` (documented ambiguity)

#### ATOMS Algorithm (atoms.py)
- Paper hyperparameters: `DELTA_PRIME = 0.1`, `M_SQUARED = 5e-4`
- `compute_psi_hat`: n=1 → `8·M²`; n>1 → Bernstein-style bound with sample std (ddof=1)
- `compute_phi_hat`: max over i∈{1..l} of `(|Δ̂_l − Δ̂_i| − ψ̂_l − ψ̂_i)₊`; l=1 always gives 0
- `atoms_pairwise_compare`: winner=1 if `delta_hat ≤ 0` (f1 better), winner=2 if `delta_hat > 0`
- `run_atoms_tournament`: pivot-based elimination; pivot wins if no challenger beats it
- `select_fixed_cv`: takes optional `candidate_specs: Dict[int, CandidateModel]`; falls back
  to Ridge(alpha=1.0) when not provided. Fold construction = sequential split (documented ambiguity).
- `val_errors_by_month` keys: integer month indices 0..T-1, most recent = T-1

#### Trading (trading.py)
- Signal rule: `sign(forecast)` — zero forecast → flat (not short); prose ambiguity resolved by formula
- Wealth: `W_t = W_{t-1} · (1 + r_t · signal_t)`, uses raw returns (not excess)
- `compute_terminal_wealth_excess`: returns `W_T_atoms / W_T_baseline − 1`

### Index Handling (models.py)
- `get_training_data` supports both `DatetimeIndex` and `PeriodIndex` via `to_period('M')`
- Uses `np.asarray()` on PeriodIndex comparisons (newer pandas returns numpy arrays directly, no `.values`)
- `window_months` windowing: finds distinct calendar months, takes last k, includes all observations

### Experiment Scripts (exp/)

All scripts follow these conventions:
- `sys.path.insert(0, <repo_root>)` at the top so `src.*` imports resolve
- `fast_mode=True` default in exp2/3/4 for quick testing (3 seeds, 5 industries, 6 candidates)
- exp2 writes `results/exp2_predictions.csv` (per-seed long-format cache); exp3 and exp4 load this before running exp2 again
- exp1 averages predictions across seeds then computes R²; exp2/3 compute R² per seed then average (per paper Section 5)

#### exp1 (`exp/exp1_complexity_tradeoff.py`)
- Compares: `ridge_64`, `rf_64`, `rf_all`
- Calls `get_training_data(X, y, t_month, window=None)` — returns `list[(X_m, y_m)]`
- Calls `subsample_month(X_m, y_m, seed=seed)` — returns `(X_sub, y_sub)` (4/5 of rows)
- Uses `sklearn.base.clone` to safely re-instantiate per-industry model
- Outputs: `results/exp1_annual_r2.csv`, `results/exp1_winner_counts.csv`
- Returns: DataFrame `[year, industry, ridge_64, rf_64, rf_all]`

#### exp2 (`exp/exp2_atoms_vs_baselines.py`)
- `build_candidate_val_errors(X, y_industry, t_month, seed, candidates)` → `{cand_idx: {month_j: sq_errors}}`
  - Calls `compute_validation_errors(candidate, X_m, y_m, seed=seed)` for each month j
- Calls `run_atoms_tournament(val_errors)`, `select_fixed_val(val_errors, window=w)`, `select_fixed_cv(val_errors)`
- All three return an integer candidate index into the `candidates` list
- After selection: calls `fit_candidate(cand, X_train, y_train)` which returns a fitted sklearn estimator
- `_get_fast_candidates()` returns first 6 entries of `CandidateModel.all_candidates()`
- Outputs: `exp2_predictions.csv`, `exp2_full_r2.csv`, `exp2_annual_r2.csv`, `exp2_summary.csv`
- Return dict keys: `methods`, `full_r2_by_industry`, `annual_r2_by_industry_method`, `predictions`

#### exp3 (`exp/exp3_recession_analysis.py`)
- Calls `get_recession_masks()` → `dict[str, tuple[str, str]]` e.g. `{'Gulf War': ('1990-06', '1990-10'), ...}`
- Calls `compute_recession_r2(y_true, y_pred, dates, start_str, end_str)` → float
- Full OOS R² computed per-seed then averaged (consistent with exp2 metric averaging)
- Output: `exp3_recession_table.csv`

#### exp4 (`exp/exp4_trading_strategy.py`)
- Averages predictions across seeds first (unlike exp3's per-seed R²), then delegates to trading.py
- Calls `compute_industry_wealth_paths(avg_preds_df, ind, present_methods)` → `{method: ndarray}`
  - avg_preds_df has columns: `[date, industry, method, y_pred, y_true]`
- Calls `compute_terminal_wealth_excess(atoms_tw, baseline_tw)` → `atoms_tw / baseline_tw - 1`
- Output: `exp4_terminal_wealth.csv` (index=industry, columns=baseline methods + Mean row)

#### run_all.py (`exp/run_all.py`)
- `os.chdir` to repo root before imports
- Runs all four experiments sequentially; exp3/4 reuse the exp2 CSV cache

### Running Tests
No test framework installed by default. Use inline Python checks:
```bash
cd /workspace/project && python -c "import sys; sys.path.insert(0,'.'); from src.models import CandidateModel; print(len(CandidateModel.all_candidates()))"
```
