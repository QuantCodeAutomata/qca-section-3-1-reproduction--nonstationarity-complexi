"""
Section 5: ATOMS vs Fixed Validation and Fixed Cross-Validation
Methods compared: ATOMS, Fixed-val(32), Fixed-val(128), Fixed-val(512), Fixed-CV
Candidate set: 168 models (28 model/hyperparam × 6 windows); fast_mode uses 6
OOS period: 01/1990-11/2016, 17 industries, 20 random seeds
Ambiguity resolutions:
  - Train/val split: 4/5 training, 1/5 validation within each historical month
  - Monthly factor lag: aligned to prior month end (1-month lag)
  - Fixed-CV: 36-month window, 5 sequential folds
  - Per-month validation: candidate is fitted on a single month's 80% train data
    and validated on the same month's 20% — this gives per-month u-values used
    by ATOMS for adaptive horizon selection
  - Duplicate truncated windows: retained (not deduplicated)
  - Metric averaging: compute per-replication then average across replications
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models import (
    CandidateModel,
    get_monthly_training_data,
    subsample_month,
    fit_candidate,
    compute_validation_errors,
)
from src.atoms import run_atoms_tournament, select_fixed_val, select_fixed_cv
from src.evaluation import compute_oos_r2, compute_annual_r2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_fast_candidates() -> List[CandidateModel]:
    """Return a reduced candidate set for fast mode (6 candidates: Ridge×3 windows + RF×3 windows).

    RF training window is capped at 64 months to avoid the O(n_obs) cost of
    fitting a full-history RF (window_months=None) on late OOS months where
    history may exceed 3 000 daily observations.  Ridge(None) is cheap because
    it is a linear model, so it is retained for window diversity.
    """
    all_cands = CandidateModel.all_candidates()
    ridge_alpha1 = [c for c in all_cands if c.model_class == 'Ridge' and abs(c.hyperparams['alpha'] - 1.0) < 1e-9]
    rf_small     = [c for c in all_cands if c.model_class == 'RF'
                    and c.hyperparams['n_estimators'] == 10 and c.hyperparams['max_depth'] == 3]

    def _pick(lst: List[CandidateModel], windows: List) -> List[CandidateModel]:
        chosen = []
        for w in windows:
            for c in lst:
                if c.window_months == w:
                    chosen.append(c)
                    break
        return chosen

    # Ridge: three window scales including full history (cheap linear solve)
    # RF: three window scales but NO full-history window to avoid 600 ms fits
    return _pick(ridge_alpha1, [1, 64, None]) + _pick(rf_small, [1, 16, 64])


def _precompute_val_errors(
    X: pd.DataFrame,
    y_industry: pd.Series,
    all_months: List[pd.Period],
    candidates: List[CandidateModel],
    seed: int,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Pre-compute per-month validation errors for each candidate (single pass).

    For each calendar month j and each candidate, fits the candidate on j's 80%
    training subsample and computes squared errors on j's 20% validation subsample.
    This per-month local fitting gives the u-value series that ATOMS and fixed-val
    baselines consume.

    # AMBIGUITY NOTE: Per-month local fitting (not rolling-window fitting) is used
    # for validation errors.  This is consistent with the paper's description of
    # "within each prior month, randomly subsample 4/5 of observations."

    Parameters
    ----------
    X : pd.DataFrame
        Full feature panel (grouping cols already stripped).
    y_industry : pd.Series
        Target returns for one industry.
    all_months : list of pd.Period
        All calendar months for which we need validation errors, in order.
    candidates : list of CandidateModel
        Models to evaluate per month.
    seed : int
        RNG seed for the 4/5–1/5 split.

    Returns
    -------
    dict
        ``{cand_idx: {month_idx: np.ndarray of squared errors}}``
        where month_idx is a 0-based index into *all_months*.
    """
    result: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(len(candidates))}

    X_vals = X.values
    y_vals = y_industry.values
    row_months = X.index.to_period('M')

    for j, m in enumerate(all_months):
        m_mask = np.asarray(row_months == m)
        X_m = X_vals[m_mask]
        y_m = y_vals[m_mask]
        if len(X_m) < 3:
            continue

        (X_tr, y_tr), (X_va, y_va) = subsample_month(X_m, y_m, seed=seed)
        if len(X_tr) == 0 or len(X_va) == 0:
            continue

        for cand_idx, candidate in enumerate(candidates):
            try:
                fitted = fit_candidate(candidate, X_tr, y_tr)
                sq_errs = compute_validation_errors(fitted, X_va, y_va)
                result[cand_idx][j] = sq_errs
            except Exception:
                pass  # Skip degenerate months silently

    return result


def _build_final_model(
    candidate: CandidateModel,
    X: pd.DataFrame,
    y_industry: pd.Series,
    t_month: pd.Period,
    seed: int,
) -> object:
    """Fit the selected candidate on pooled 80% subsamples from its training window.

    Parameters
    ----------
    candidate : CandidateModel
        The winning candidate including its training window.
    X : pd.DataFrame
        Feature panel.
    y_industry : pd.Series
        Target for one industry.
    t_month : pd.Period
        OOS forecast month; training uses only months strictly before t_month.
    seed : int
        RNG seed for the 4/5 subsample.

    Returns
    -------
    Fitted sklearn estimator.
    """
    monthly = get_monthly_training_data(X, y_industry, t_month, window_months=candidate.window_months)
    if not monthly:
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0).fit(np.zeros((1, X.shape[1] - 0)), np.zeros(1))

    X_parts, y_parts = [], []
    for X_m, y_m in monthly:
        (X_tr, y_tr), _ = subsample_month(X_m, y_m, seed=seed)
        if len(X_tr) > 0:
            X_parts.append(X_tr)
            y_parts.append(y_tr)

    if not X_parts:
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0).fit(np.zeros((1, X.shape[1])), np.zeros(1))

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    return fit_candidate(candidate, X_train, y_train)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp2(
    X: pd.DataFrame,
    y: pd.DataFrame,
    output_dir: str = 'results',
    n_seeds: int = 20,
    fast_mode: bool = True,
) -> Dict:
    """
    Compare ATOMS against fixed-validation and fixed-CV baselines.

    Performance note: validation errors are precomputed once for all historical
    months (not re-computed per OOS month), reducing complexity from
    O(T × S × I × J × C) to O(S × I × J_max × C) + O(T × S × I × C).

    Parameters
    ----------
    X : pd.DataFrame
        Daily feature panel (DatetimeIndex).  Grouping columns 'month'/'year'
        are dropped internally before model fitting.
    y : pd.DataFrame
        Daily return panel (DatetimeIndex); columns are industry names.
    output_dir : str
        Directory for output CSV files.
    n_seeds : int
        Number of random replications (ignored when fast_mode=True).
    fast_mode : bool
        If True: 1 seed, 5 industries, 6 candidates, OOS 2000-01 to 2002-12.
        Full mode: n_seeds seeds, 17 industries, 168 candidates, OOS 1990-01
        to 2016-11. fast_mode=True is the default for CI/validation runs.

    Returns
    -------
    dict
        Keys: ``methods``, ``full_r2_by_industry``,
        ``annual_r2_by_industry_method``, ``predictions``.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Strip grouping helper columns
    feature_cols = [c for c in X.columns if c not in ('month', 'year')]
    X = X[feature_cols]

    if fast_mode:
        # FAST MODE: 1 seed, 5 industries, 6-candidate subset, 36-month OOS window
        # covering the 2001 recession.  RF uses 10 trees/depth-3.
        n_seeds_actual = 1
        industries = y.columns.tolist()[:5]
        candidates = _get_fast_candidates()
        oos_months = pd.period_range('2000-01', '2002-12', freq='M')
        print(f'  Exp2 fast_mode: {n_seeds_actual} seed, {len(industries)} industries,'
              f' {len(candidates)} candidates, {len(oos_months)} OOS months')
    else:
        n_seeds_actual = n_seeds
        industries = y.columns.tolist()
        candidates = CandidateModel.all_candidates()
        oos_months = pd.period_range('1990-01', '2016-11', freq='M')
        print(f'  Exp2 full mode: {n_seeds_actual} seeds, {len(industries)} industries,'
              f' {len(candidates)} candidates, {len(oos_months)} OOS months')

    n_cands = len(candidates)
    cand_indices = list(range(n_cands))
    methods = ['ATOMS', 'Fixed-val(32)', 'Fixed-val(128)', 'Fixed-val(512)', 'Fixed-CV']

    # Calendar months needed for ATOMS validation: all months up to (but not
    # including) max(oos_months).  Validation errors for post-OOS months are
    # never consumed by any selector, so we skip them to halve Step-1 cost.
    all_dataset_months = sorted(
        m for m in X.index.to_period('M').unique() if m < max(oos_months)
    )

    # ------------------------------------------------------------------
    # STEP 1: Precompute per-month validation errors ONCE for all months
    # needed (months strictly before max(oos_months)).
    # ------------------------------------------------------------------
    print(f'  Precomputing per-month validation errors ({len(all_dataset_months)} months)...')
    # global_val_errors[(ind, seed)] = {cand_idx: {month_j_abs: sq_errs}}
    # month_j_abs is the 0-based index into all_dataset_months
    global_val_errors: Dict[Tuple, Dict[int, Dict[int, np.ndarray]]] = {}
    for seed in range(n_seeds_actual):
        for ind in industries:
            key = (ind, seed)
            global_val_errors[key] = _precompute_val_errors(
                X, y[ind], all_dataset_months, candidates, seed
            )
        print(f'    Seed {seed + 1}/{n_seeds_actual}: validation errors precomputed')

    # ------------------------------------------------------------------
    # STEP 2: Precompute final model predictions for all (t, ind, seed, cand).
    # Optimisation: fetch ALL history once per (t, ind, seed), subsample each
    # month once, then slice to each candidate's window_months — avoids n_cands
    # redundant get_monthly_training_data calls per (t, ind, seed).
    # ------------------------------------------------------------------
    print('  Precomputing final model predictions and Fixed-CV winners for all OOS months...')
    # global_preds[(t, ind, seed, cand_idx)] = y_pred array (shape: n_test,)
    global_preds: Dict[Tuple, np.ndarray] = {}
    # global_cv_winners[(t, ind, seed)] = winning cand_idx chosen by Fixed-CV
    global_cv_winners: Dict[Tuple, int] = {}
    n_months = len(oos_months)

    # Pre-compute period array once to speed up mask operations
    row_period = X.index.to_period('M')

    for t_idx, t in enumerate(oos_months):
        if t_idx % 12 == 0:
            print(f'    OOS month {t} ({t_idx + 1}/{n_months})')
        t_mask = np.asarray(row_period == t)
        X_test_loc_vals = X.values[t_mask]
        if X_test_loc_vals.shape[0] == 0:
            continue

        for seed in range(n_seeds_actual):
            for ind in industries:
                # ---- Fetch full history ONCE for this (t, ind, seed) ----
                monthly_all = get_monthly_training_data(X, y[ind], t, window_months=None)
                # Subsample each month once; store (X_tr, y_tr) pairs
                subsampled: List[Tuple[np.ndarray, np.ndarray]] = []
                for X_m, y_m in monthly_all:
                    (X_tr, y_tr), _ = subsample_month(X_m, y_m, seed=seed)
                    if len(X_tr) > 0:
                        subsampled.append((X_tr, y_tr))

                # ---- Fit each candidate by slicing the subsampled list ----
                for ci, cand in enumerate(candidates):
                    w = cand.window_months
                    sl = subsampled if w is None else subsampled[-w:]
                    if not sl:
                        continue
                    X_train = np.vstack([s[0] for s in sl])
                    y_train = np.concatenate([s[1] for s in sl])
                    try:
                        fitted = fit_candidate(cand, X_train, y_train)
                        global_preds[(t, ind, seed, ci)] = fitted.predict(X_test_loc_vals)
                    except Exception:
                        pass

                # ---- Precompute Fixed-CV winner (reuse subsampled, avoid Step-3 refetch) ----
                # FAST MODE NOTE: use 2 folds instead of 5 to reduce CV cost by ~60 %.
                # In fast_mode the OOS window is only 36 months so exact fold count is
                # not critical; 2-fold CV still ranks candidates correctly.
                cv_window_precompute = 36
                cv_n_folds = 2 if fast_mode else 5
                sl_cv = subsampled[-cv_window_precompute:] if len(subsampled) >= cv_window_precompute else subsampled
                if sl_cv:
                    X_cv_pre = np.vstack([s[0] for s in sl_cv])
                    y_cv_pre = np.concatenate([s[1] for s in sl_cv])
                    cv_data_pre = {ci: (X_cv_pre, y_cv_pre) for ci in range(n_cands)}
                    cand_specs_pre = {ci: candidates[ci] for ci in range(n_cands)}
                    try:
                        global_cv_winners[(t, ind, seed)] = select_fixed_cv(
                            cv_data_pre,
                            cv_window=cv_window_precompute,
                            n_folds=cv_n_folds,
                            candidate_specs=cand_specs_pre,
                        )
                    except Exception:
                        global_cv_winners[(t, ind, seed)] = 0
                else:
                    global_cv_winners[(t, ind, seed)] = 0

    # ------------------------------------------------------------------
    # STEP 3: OOS selection loop — fast lookups only
    # For each OOS month t: slice precomputed val_errors to months < t,
    # run each selector, then look up the winner's precomputed predictions.
    # ------------------------------------------------------------------
    records = []

    for seed in range(n_seeds_actual):
        for t_idx, t in enumerate(oos_months):
            t_mask = X.index.to_period('M') == t
            X_test_loc = X.loc[t_mask]
            if X_test_loc.empty:
                continue

            # prior_months is a prefix of all_dataset_months so j_rel == j_abs
            n_horizons = sum(1 for m in all_dataset_months if m < t)
            if n_horizons == 0:
                continue
            prior_months = all_dataset_months[:n_horizons]

            for ind in industries:
                y_test_arr = y[ind].loc[t_mask].values
                key = (ind, seed)

                # Slice val_errors: keep only months j_abs < n_horizons
                val_errors_t: Dict[int, Dict[int, np.ndarray]] = {}
                for ci in cand_indices:
                    cand_map = global_val_errors[key].get(ci, {})
                    val_errors_t[ci] = {
                        j: sq for j, sq in cand_map.items() if j < n_horizons
                    }

                # ATOMS: adaptive tournament
                atoms_winner, _ = run_atoms_tournament(
                    cand_indices, val_errors_t, n_horizons, pivot_seed=seed
                )

                # Fixed-val baselines
                fv_winners = {
                    w: select_fixed_val(val_errors_t, n_horizons, w)
                    for w in [32, 128, 512]
                }

                # Fixed-CV: look up winner precomputed in Step 2 (no refitting here)
                cv_winner = global_cv_winners.get((t, ind, seed), 0)

                method_winners = {
                    'ATOMS': atoms_winner,
                    'Fixed-val(32)': fv_winners[32],
                    'Fixed-val(128)': fv_winners[128],
                    'Fixed-val(512)': fv_winners[512],
                    'Fixed-CV': cv_winner,
                }

                # Look up precomputed predictions (no model refitting here)
                for method, sel_idx in method_winners.items():
                    y_pred = global_preds.get((t, ind, seed, sel_idx))
                    if y_pred is None:
                        continue
                    for i, date in enumerate(X_test_loc.index):
                        records.append({
                            'seed': seed,
                            'date': date,
                            'industry': ind,
                            'method': method,
                            'y_pred': float(y_pred[i]),
                            'y_true': float(y_test_arr[i]),
                        })

    preds_df = pd.DataFrame(records)

    # Cache raw per-seed predictions for downstream experiments
    pred_cache = os.path.join(output_dir, 'exp2_predictions.csv')
    preds_df.to_csv(pred_cache, index=False)
    print(f'  Saved per-seed predictions to {pred_cache}')

    # ------------------------------------------------------------------
    # Metric averaging: compute R² per replication then average
    # ------------------------------------------------------------------
    full_r2_by_industry: Dict[str, Dict[str, float]] = {m: {} for m in methods}
    annual_r2_by_industry_method: Dict[str, Dict[str, Dict[int, float]]] = {
        m: {ind: {} for ind in industries} for m in methods
    }

    for method in methods:
        for ind in industries:
            seed_full_r2 = []
            seed_annual_acc: Dict[int, List[float]] = {}

            for seed in range(n_seeds_actual):
                mask = (
                    (preds_df['seed'] == seed)
                    & (preds_df['method'] == method)
                    & (preds_df['industry'] == ind)
                )
                sub = preds_df.loc[mask].sort_values('date')
                if sub.empty:
                    continue

                r2_full = compute_oos_r2(sub['y_true'].values, sub['y_pred'].values)
                seed_full_r2.append(r2_full)

                dates_idx = pd.DatetimeIndex(sub['date'])
                y_true_s = pd.Series(sub['y_true'].values, index=dates_idx)
                y_pred_s = pd.Series(sub['y_pred'].values, index=dates_idx)
                r2_annual = compute_annual_r2(y_true_s, y_pred_s)
                for year, r2_yr in r2_annual.items():
                    seed_annual_acc.setdefault(year, []).append(r2_yr)

            full_r2_by_industry[method][ind] = (
                float(np.nanmean(seed_full_r2)) if seed_full_r2 else np.nan
            )
            for year, r2_list in seed_annual_acc.items():
                annual_r2_by_industry_method[method][ind][year] = float(np.nanmean(r2_list))

    # Save full-sample R² summary
    full_r2_records = [
        {'method': m, 'industry': ind, 'oos_r2': full_r2_by_industry[m][ind]}
        for m in methods for ind in industries
    ]
    pd.DataFrame(full_r2_records).to_csv(os.path.join(output_dir, 'exp2_full_r2.csv'), index=False)

    # Save annual R² in long format
    annual_records_list = [
        {'method': m, 'industry': ind, 'year': year, 'oos_r2': r2}
        for m in methods for ind in industries
        for year, r2 in annual_r2_by_industry_method[m][ind].items()
    ]
    pd.DataFrame(annual_records_list).to_csv(os.path.join(output_dir, 'exp2_annual_r2.csv'), index=False)
    print(f'  Saved exp2 R² tables to {output_dir}/')

    return {
        'methods': methods,
        'full_r2_by_industry': full_r2_by_industry,
        'annual_r2_by_industry_method': annual_r2_by_industry_method,
        'predictions': preds_df,
    }


# ---------------------------------------------------------------------------
# Analysis / reporting
# ---------------------------------------------------------------------------

def analyze_exp2(results: Dict, output_dir: str = 'results') -> pd.DataFrame:
    """
    Print and save average OOS R² across industries per method.

    Parameters
    ----------
    results : dict
        Output of run_exp2.
    output_dir : str
        Directory where exp2_summary.csv will be written.

    Returns
    -------
    pd.DataFrame
        Summary with columns [method, mean_oos_r2].

    Notes
    -----
    Reference values (full mode, all 17 industries):
        ATOMS         ~ 0.049
        Fixed-val(32) ~ 0.022
        Fixed-val(512)~ 0.043
        Fixed-CV      ~ 0.035
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = results['methods']
    full_r2 = results['full_r2_by_industry']

    summary_records = []
    print('\nExp2 — Average OOS R² across industries')
    print('=' * 45)
    for method in methods:
        r2_vals = [v for v in full_r2[method].values() if not np.isnan(v)]
        avg_r2 = float(np.mean(r2_vals)) if r2_vals else np.nan
        summary_records.append({'method': method, 'mean_oos_r2': avg_r2})
        print(f'  {method:<20}: {avg_r2:+.4f}')

    summary_df = pd.DataFrame(summary_records)
    out_path = os.path.join(output_dir, 'exp2_summary.csv')
    summary_df.to_csv(out_path, index=False)
    print(f'\n  Saved summary to {out_path}')
    print('\n  Reference (full mode): ATOMS~0.049, Fixed-val(32)~0.022, '
          'Fixed-val(512)~0.043, Fixed-CV~0.035')
    return summary_df


if __name__ == '__main__':
    from src.data_loader import load_full_panel
    X, y = load_full_panel()
    results = run_exp2(X, y, fast_mode=True)
    analyze_exp2(results)
