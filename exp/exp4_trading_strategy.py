"""
Sign-based trading strategy evaluation using ATOMS vs baseline method predictions.

Ambiguity resolutions:
  - Wealth uses raw returns (not excess returns) per compute_wealth_path.
  - sign(0) = 0 (flat), per formula rule; see trading.py AMBIGUITY NOTE.
  - Wealth per replication is computed from seed-averaged forecasts rather than
    averaging terminal wealth across seeds, to match the predictive metric
    averaging logic used in exp2/exp3.
  - Transaction costs, slippage, short-sale constraints, and turnover are omitted
    (consistent with the paper's core trading experiment).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.trading import (
    compute_trading_signals,
    compute_wealth_path,
    compute_terminal_wealth_excess,
    compute_industry_wealth_paths,
)
from exp.exp2_atoms_vs_baselines import run_exp2


def _build_method_panel(
    avg_preds: pd.DataFrame,
    method: str,
    industries: list,
) -> tuple:
    """
    Build aligned (forecasts_df, returns_df) DataFrames for a given method.

    Parameters
    ----------
    avg_preds : pd.DataFrame
        Seed-averaged predictions; columns [date, industry, method, y_pred, y_true].
    method : str
        Method name to filter on.
    industries : list
        Ordered list of industry names.

    Returns
    -------
    (forecasts_df, returns_df) : tuple of pd.DataFrame
        Both have DatetimeIndex and industry columns, aligned on the union of dates.
    """
    method_df = avg_preds[avg_preds['method'] == method].sort_values('date')

    forecasts_df = (
        method_df.pivot(index='date', columns='industry', values='y_pred')
        .reindex(columns=industries)
    )
    returns_df = (
        method_df.pivot(index='date', columns='industry', values='y_true')
        .reindex(columns=industries)
    )
    forecasts_df.index = pd.DatetimeIndex(forecasts_df.index)
    returns_df.index = pd.DatetimeIndex(returns_df.index)
    return forecasts_df, returns_df


def run_exp4(
    X: pd.DataFrame,
    y: pd.DataFrame,
    output_dir: str = 'results',
    fast_mode: bool = True,
) -> pd.DataFrame:
    """
    Evaluate sign-based long/short trading strategies derived from each method.

    For each method and industry, the forecast sign determines position direction
    (+1 long, -1 short, 0 flat).  Terminal wealth ratios of ATOMS relative to
    each baseline are computed and averaged across industries.

    Parameters
    ----------
    X : pd.DataFrame
        Daily feature panel (DatetimeIndex).
    y : pd.DataFrame
        Daily return panel (DatetimeIndex); columns are industry names.
    output_dir : str
        Directory for output CSV and for the exp2 prediction cache.
    fast_mode : bool
        Passed through to run_exp2 when predictions must be generated.

    Returns
    -------
    pd.DataFrame
        Terminal wealth excess ratios (ATOMS / baseline − 1), with industries
        as the index and baseline methods as columns; final row is the
        cross-industry mean.

    Notes
    -----
    Expected approximate excess ratios (full mode, averaged across 17 industries):
        over Fixed-val(32)  : +3.38
        over Fixed-val(128) : +0.48
        over Fixed-val(512) : +0.31
        over Fixed-CV       : +3.54
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load or generate per-seed predictions from exp2
    pred_cache = os.path.join(output_dir, 'exp2_predictions.csv')
    if os.path.exists(pred_cache):
        print(f'  Loading cached predictions from {pred_cache}')
        preds_df = pd.read_csv(pred_cache, parse_dates=['date'])
    else:
        print('  No cached predictions found; running exp2...')
        results2 = run_exp2(X, y, output_dir=output_dir, fast_mode=fast_mode)
        preds_df = results2['predictions']

    all_methods = ['ATOMS', 'Fixed-val(32)', 'Fixed-val(128)', 'Fixed-val(512)', 'Fixed-CV']
    present_methods = [m for m in all_methods if m in preds_df['method'].unique()]
    baseline_methods = [m for m in present_methods if m != 'ATOMS']
    industries = sorted(preds_df['industry'].unique().tolist())

    if 'ATOMS' not in present_methods:
        raise ValueError('ATOMS predictions not found; cannot compute excess ratios.')

    # Average predictions across seeds → one forecast per (date, industry, method)
    avg_preds = (
        preds_df
        .groupby(['date', 'industry', 'method'], as_index=False)
        .agg(y_pred=('y_pred', 'mean'), y_true=('y_true', 'first'))
    )

    # Compute wealth paths per method: {industry: pd.Series}
    method_wealth: dict[str, dict[str, pd.Series]] = {}
    for method in present_methods:
        forecasts_df, returns_df = _build_method_panel(avg_preds, method, industries)
        # Drop any all-NaN columns or rows to avoid wealth multiplication issues
        forecasts_df = forecasts_df.dropna(how='all', axis=0).fillna(0.0)
        returns_df = returns_df.reindex(forecasts_df.index).fillna(0.0)
        # compute_industry_wealth_paths: (forecasts_df, returns_df) → {industry: pd.Series}
        method_wealth[method] = compute_industry_wealth_paths(forecasts_df, returns_df)

    # Terminal wealth per method per industry
    terminal_wealth: dict[str, dict[str, float]] = {m: {} for m in present_methods}
    for method in present_methods:
        for ind in industries:
            wp = method_wealth[method].get(ind)
            terminal_wealth[method][ind] = float(wp.iloc[-1]) if wp is not None else np.nan

    # Terminal wealth excess: ATOMS / baseline - 1 per industry
    # compute_terminal_wealth_excess takes pd.Series (wealth paths), not scalars.
    excess_records = []
    for ind in industries:
        row = {'industry': ind}
        atoms_wp = method_wealth['ATOMS'].get(ind)
        for baseline in baseline_methods:
            base_wp = method_wealth[baseline].get(ind)
            if atoms_wp is not None and base_wp is not None:
                excess = compute_terminal_wealth_excess(atoms_wp, base_wp)
            else:
                excess = np.nan
            row[baseline] = excess
        excess_records.append(row)

    excess_df = pd.DataFrame(excess_records).set_index('industry')

    # Append cross-industry mean row
    mean_row = excess_df.mean().to_frame().T
    mean_row.index = pd.Index(['Mean'])
    excess_df = pd.concat([excess_df, mean_row])

    out_path = os.path.join(output_dir, 'exp4_terminal_wealth.csv')
    excess_df.to_csv(out_path)
    print(f'  Saved terminal wealth excess ratios to {out_path}')

    print('\nExp4 — Terminal Wealth Excess (ATOMS / baseline − 1), mean across industries')
    print('=' * 65)
    mean_excess = excess_df.loc['Mean']
    for baseline in baseline_methods:
        print(f'  ATOMS over {baseline:<22}: {mean_excess[baseline]:+.4f}')
    print('\n  Reference (full mode): over Fixed-val(32): ~3.38, '
          'over Fixed-val(128): ~0.48, over Fixed-val(512): ~0.31, '
          'over Fixed-CV: ~3.54')

    return excess_df


if __name__ == '__main__':
    from src.data_loader import load_full_panel
    X, y = load_full_panel()
    result = run_exp4(X, y, fast_mode=True)
    print('\nTerminal wealth excess ratios by industry:')
    print(result.to_string())
