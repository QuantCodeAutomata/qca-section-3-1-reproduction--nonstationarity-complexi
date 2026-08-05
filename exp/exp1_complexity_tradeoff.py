"""
Section 3.1 Reproduction: Nonstationarity-Complexity Tradeoff
Compares 3 fixed models:
  (a) Ridge, alpha=1, 64-month training window
  (b) Random Forest, n_estimators=200, max_depth=5, 64-month window
  (c) Random Forest, n_estimators=200, max_depth=5, ALL historical data
Evaluation: annual OOS R² per industry, 17 industries, OOS period 01/1990-11/2016
Ambiguity resolutions:
  - Industry targets: daily raw returns (not excess)
  - Monthly factors aligned with 1-month lag to avoid lookahead
  - Size filter: appendix specification (bottom 25% ME) would apply for Gu chars;
    here synthetic proxies are used (CRSP/Gu data unavailable without WRDS access)
  - Tie-breaking: in case of tie, alphabetical model name order (ridge < rf)
  - 20 random seeds for 4/5 monthly subsampling
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from src.models import get_monthly_training_data, subsample_month, fit_candidate
from src.evaluation import compute_oos_r2, compute_annual_r2


def run_exp1(
    X: pd.DataFrame,
    y: pd.DataFrame,
    output_dir: str = 'results',
    fast_mode: bool = False,
) -> pd.DataFrame:
    """
    Run Experiment 1: compare Ridge-64, RF-64, and RF-all on OOS annual R².

    Parameters
    ----------
    X : pd.DataFrame
        Daily feature panel with DatetimeIndex.  The 'month' and 'year' grouping
        columns produced by ``build_predictor_panel`` are dropped internally so
        only the 129 predictor features are passed to estimators.
    y : pd.DataFrame
        Daily return panel with DatetimeIndex; columns are industry names.
    output_dir : str
        Directory where exp1_annual_r2.csv will be written.
    fast_mode : bool
        If True, use a reduced configuration to complete in ~3 minutes:
        OOS 2000-01 to 2002-12, 1 seed, 5 industries, RF(10 trees, max_depth=3).
        Paper spec (full mode): OOS 1990-01 to 2016-11, 20 seeds, all 17 industries,
        RF(200 trees, max_depth=5).  fast_mode=True is for validation/CI use.

    Returns
    -------
    pd.DataFrame
        Columns [year, industry, ridge_64, rf_64, rf_all] with annual OOS R².
    """
    os.makedirs(output_dir, exist_ok=True)

    # Drop grouping helpers added by build_predictor_panel if present
    feature_cols = [c for c in X.columns if c not in ('month', 'year')]
    X = X[feature_cols]

    if fast_mode:
        # FAST MODE: reduced parameters for quick validation
        # NOTE: RF uses 10 trees / max_depth=3 instead of paper's 200/5
        oos_months = pd.period_range('2000-01', '2002-12', freq='M')
        industries = y.columns.tolist()[:5]
        seeds = [0]
        rf_kwargs = {'n_estimators': 10, 'max_depth': 3, 'random_state': 0}
        print(f'  Exp1 fast_mode: {len(seeds)} seed, {len(industries)} industries,'
              f' {len(oos_months)} OOS months, RF(10,3)')
    else:
        oos_months = pd.period_range('1990-01', '2016-11', freq='M')
        industries = y.columns.tolist()
        seeds = list(range(20))
        rf_kwargs = {'n_estimators': 200, 'max_depth': 5, 'random_state': 0}
        print(f'  Exp1 full mode: {len(seeds)} seeds, {len(industries)} industries,'
              f' {len(oos_months)} OOS months, RF(200,5)')

    # Model specifications — paper Section 3.1 exact hyperparameters in full mode
    models_spec = {
        'ridge_64': {'cls': Ridge,  'kwargs': {'alpha': 1},  'window': 64},
        'rf_64':    {'cls': RandomForestRegressor, 'kwargs': rf_kwargs, 'window': 64},
        'rf_all':   {'cls': RandomForestRegressor, 'kwargs': rf_kwargs, 'window': None},
    }
    model_names = list(models_spec.keys())

    records = []
    n_months = len(oos_months)

    for t_idx, t in enumerate(oos_months):
        if t_idx % 12 == 0:
            print(f'  Exp1: processing {t} ({t_idx + 1}/{n_months})')

        t_mask = X.index.to_period('M') == t
        X_test = X.loc[t_mask]
        if X_test.empty:
            continue
        X_test_arr = X_test.values
        y_test = y.loc[t_mask]

        # Pre-fetch full monthly panel once per industry, then slice by window.
        # This avoids re-scanning the full date index for each (model, industry) pair
        # (15 redundant scans → 5 scans, one per industry).
        monthly_by_ind: dict = {}
        for ind in industries:
            monthly_by_ind[ind] = get_monthly_training_data(X, y[ind], t, window_months=None)

        n_prior = len(monthly_by_ind[industries[0]])
        if n_prior == 0:
            continue

        n_test = len(X_test_arr)
        # Accumulate per-seed, per-model, per-industry predictions
        seed_preds = {
            name: {ind: np.zeros((len(seeds), n_test)) for ind in industries}
            for name in model_names
        }

        for s_idx, seed in enumerate(seeds):
            # Precompute per-month, per-industry subsampled (X_tr, y_tr) tuples;
            # this is shared across all 3 models so subsampling is performed once
            # per (seed, industry, month) rather than once per (seed, model, industry, month).
            subsampled_by_ind: dict = {}
            for ind in industries:
                subsampled_by_ind[ind] = []
                for X_m, y_m in monthly_by_ind[ind]:
                    (X_tr, y_tr), _ = subsample_month(X_m, y_m, seed=seed)
                    subsampled_by_ind[ind].append((X_tr, y_tr) if len(X_tr) > 0 else None)

            for name, spec in models_spec.items():
                window = spec['window']
                for ind in industries:
                    full_sub = subsampled_by_ind[ind]
                    # Slice to requested window (None → use all)
                    sliced = full_sub if window is None else full_sub[-window:]
                    valid = [p for p in sliced if p is not None]
                    if not valid:
                        continue

                    X_train = np.vstack([p[0] for p in valid])
                    y_train = np.concatenate([p[1] for p in valid])

                    model = spec['cls'](**spec['kwargs'])
                    model.fit(X_train, y_train)
                    seed_preds[name][ind][s_idx] = model.predict(X_test_arr)

        # Average predictions across seeds then store records
        for name in model_names:
            for ind in industries:
                y_pred_avg = seed_preds[name][ind].mean(axis=0)
                y_true_arr = y_test[ind].values
                for i, date in enumerate(X_test.index):
                    records.append({
                        'date': date,
                        'industry': ind,
                        'model': name,
                        'y_pred': float(y_pred_avg[i]),
                        'y_true': float(y_true_arr[i]),
                    })

    preds_df = pd.DataFrame(records)

    # Compute annual OOS R² per model per industry using evaluation.compute_annual_r2
    annual_records = []
    for name in model_names:
        for ind in industries:
            mask = (preds_df['model'] == name) & (preds_df['industry'] == ind)
            sub = preds_df.loc[mask].sort_values('date')
            if sub.empty:
                continue
            # Build pd.Series with DatetimeIndex as required by compute_annual_r2
            dates_idx = pd.DatetimeIndex(sub['date'])
            y_true_s = pd.Series(sub['y_true'].values, index=dates_idx)
            y_pred_s = pd.Series(sub['y_pred'].values, index=dates_idx)
            r2_series = compute_annual_r2(y_true_s, y_pred_s)  # returns pd.Series keyed by year
            for year, r2 in r2_series.items():
                annual_records.append({'year': year, 'industry': ind, 'model': name, 'r2': r2})

    annual_df = pd.DataFrame(annual_records)
    results_df = annual_df.pivot_table(
        index=['year', 'industry'], columns='model', values='r2', aggfunc='first'
    ).reset_index()
    results_df.columns.name = None

    for col in model_names:
        if col not in results_df.columns:
            results_df[col] = np.nan
    results_df = results_df[['year', 'industry'] + model_names]

    out_path = os.path.join(output_dir, 'exp1_annual_r2.csv')
    results_df.to_csv(out_path, index=False)
    print(f'  Saved annual R² to {out_path}')
    return results_df


def analyze_exp1(results_df: pd.DataFrame, output_dir: str = 'results') -> pd.DataFrame:
    """
    Summarise Experiment 1 results: per-year model winner counts and recession flags.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_exp1; columns [year, industry, ridge_64, rf_64, rf_all].
    output_dir : str
        Directory where exp1_winner_counts.csv will be written.

    Returns
    -------
    pd.DataFrame
        Winner counts indexed by year, one column per model.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_cols = [c for c in ['ridge_64', 'rf_64', 'rf_all'] if c in results_df.columns]
    results_df = results_df.copy()
    # idxmax — alphabetical tie-breaking because 'ridge_64' < 'rf_64' < 'rf_all'
    results_df['winner'] = results_df[model_cols].idxmax(axis=1)

    winner_counts = (
        results_df.groupby(['year', 'winner'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=model_cols, fill_value=0)
    )
    out_path = os.path.join(output_dir, 'exp1_winner_counts.csv')
    winner_counts.to_csv(out_path)
    print(f'  Saved winner counts to {out_path}')

    recession_years = {1990, 2001, 2007, 2008, 2009}  # NBER overlapping OOS years
    print('\nAnnual OOS R² Summary (Experiment 1) — mean across industries')
    print('=' * 65)
    for year in sorted(results_df['year'].unique()):
        year_data = results_df[results_df['year'] == year]
        mean_r2 = year_data[model_cols].mean()
        rec_flag = '  [RECESSION]' if year in recession_years else ''
        print(f'{year}{rec_flag}')
        for col in model_cols:
            print(f'    {col:<12}: {mean_r2[col]:+.4f}')

    print('\nWinner counts per year (number of industries won):')
    print(winner_counts.to_string())
    return winner_counts


if __name__ == '__main__':
    from src.data_loader import load_full_panel
    X, y = load_full_panel()
    results = run_exp1(X, y)
    analyze_exp1(results)
