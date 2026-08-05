"""
Evaluation metrics for out-of-sample (OOS) predictive accuracy.

All R² figures follow the paper's zero-benchmark convention:
    R² = 1 - SSE / sum(y²)
where the denominator uses raw squared actuals, *not* demeaned variance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def compute_oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the paper's zero-benchmark OOS R².

    Unlike the classical coefficient of determination, the denominator is the
    sum of squared *actual* values (not the variance of actuals), making the
    zero forecast the natural benchmark.

    Parameters
    ----------
    y_true:
        Array of realised target values.
    y_pred:
        Array of model predictions aligned with y_true.

    Returns
    -------
    float
        OOS R² = 1 - SSE / sum(y_true²).  Can be negative when the model
        performs worse than the zero benchmark.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sse = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum(y_true ** 2)
    return float(1.0 - sse / ss_total)


def compute_annual_r2(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Compute OOS R² broken down by calendar year.

    Parameters
    ----------
    y_true:
        Realised values with a DatetimeIndex or PeriodIndex that carries year
        information.
    y_pred:
        Model predictions with the same index as y_true.

    Returns
    -------
    pd.Series
        OOS R² for each year, indexed by integer year.  Only years that appear
        in the index are included.
    """
    years = y_true.index.year
    annual: Dict[int, float] = {}
    for yr in sorted(set(years)):
        mask = years == yr
        annual[yr] = compute_oos_r2(y_true.values[mask], y_pred.values[mask])
    return pd.Series(annual, dtype=float)


def compute_recession_r2(
    y_true: pd.Series,
    y_pred: pd.Series,
    recession_masks: Dict[str, pd.Series],
) -> Dict[str, float]:
    """Compute OOS R² for the full OOS period and each named recession window.

    Parameters
    ----------
    y_true:
        Realised values.
    y_pred:
        Model predictions aligned with y_true.
    recession_masks:
        Dictionary mapping recession label to a boolean pd.Series with the
        same index as y_true/y_pred.  Expected keys are ``'gulf_war'``,
        ``'dot_com'``, and ``'gfc'``; any additional keys are also handled.

    Returns
    -------
    dict
        ``{'full_oos': float, 'gulf_war': float, 'dot_com': float,
        'gfc': float}``.  Each value is the OOS R² over that sub-period.
    """
    results: Dict[str, float] = {
        "full_oos": compute_oos_r2(y_true.values, y_pred.values)
    }
    for label, mask in recession_masks.items():
        results[label] = compute_oos_r2(
            y_true.values[mask.values], y_pred.values[mask.values]
        )
    return results


def aggregate_metrics_across_seeds(metrics_by_seed: List[Dict]) -> Dict:
    """Average metric dictionaries across multiple random seeds.

    Supports one level of nesting: if a value is itself a dict (e.g. annual R²
    keyed by year), each inner value is averaged independently.

    Parameters
    ----------
    metrics_by_seed:
        List of metric dictionaries, one per random seed.  All dicts must
        share the same top-level keys; nested dicts must share the same inner
        keys across seeds.

    Returns
    -------
    dict
        Dictionary with the same structure as a single seed's metrics, with
        every scalar value replaced by its mean across seeds.
    """
    if not metrics_by_seed:
        return {}

    result: Dict = {}
    for key in metrics_by_seed[0]:
        values = [m[key] for m in metrics_by_seed]
        if isinstance(values[0], dict):
            inner_keys = values[0].keys()
            result[key] = {
                ik: float(np.mean([v[ik] for v in values])) for ik in inner_keys
            }
        else:
            result[key] = float(np.mean(values))
    return result
