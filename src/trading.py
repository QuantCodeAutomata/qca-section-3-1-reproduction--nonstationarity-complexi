"""
Trading strategy utilities.

Translates return forecasts into sign-based trading positions and accumulates
wealth paths from those positions.

Sign rule
---------
* signal = +1 when forecast > 0 (long)
* signal = −1 when forecast < 0 (short)
* signal =  0 when forecast = 0 (flat)

The zero-signal case is documented as an ambiguity resolution: the paper's
prose indicates shorting when the forecast is *non-positive*, but the formula
uses ``sign(f_t)`` which maps exactly zero to zero.  The formula takes
precedence here.

Wealth accumulation
-------------------
    W_t = W_{t−1} · (1 + r_t · signal_t)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_trading_signals(forecasts: pd.Series) -> pd.Series:
    """Convert return forecasts to sign-based long/short/flat positions.

    Parameters
    ----------
    forecasts:
        Series of scalar return forecasts; any datetime-compatible index.

    Returns
    -------
    pd.Series
        +1 (long), −1 (short), or 0 (flat) for each forecast period.
        Same index and name as *forecasts*.

    Notes
    -----
    # AMBIGUITY NOTE: sign(0) = 0 per formula; prose says short if
    # nonpositive — using formula sign rule.
    """
    signals = np.sign(forecasts.values)
    return pd.Series(signals, index=forecasts.index, name=forecasts.name, dtype=float)


def compute_wealth_path(
    signals: pd.Series,
    returns: pd.Series,
    initial_wealth: float = 1.0,
) -> pd.Series:
    """Accumulate wealth from a sequence of sign-based positions.

    Applies the recurrence

        W_t = W_{t−1} · (1 + r_t · signal_t)

    starting from *initial_wealth* at time 0 (before the first observed
    return).  The returned series therefore starts at time 1.

    Parameters
    ----------
    signals:
        Series of trading signals (typically +1 / −1 / 0) aligned with
        *returns*.
    returns:
        Series of raw period returns, same index as *signals*.
    initial_wealth:
        Starting portfolio value (default 1.0).

    Returns
    -------
    pd.Series
        Cumulative wealth at each period in the shared index.  The final
        value W_T is the terminal portfolio value.

    Notes
    -----
    # AMBIGUITY NOTE: Wealth uses raw returns (not excess returns) for
    # economic interpretation.
    """
    growth = 1.0 + signals.values * returns.values
    wealth = initial_wealth * np.cumprod(growth)
    return pd.Series(wealth, index=returns.index, dtype=float)


def compute_terminal_wealth_excess(
    wealth_atoms: pd.Series,
    wealth_baseline: pd.Series,
) -> float:
    """Compute terminal wealth of ATOMS relative to a baseline strategy.

    Parameters
    ----------
    wealth_atoms:
        Wealth path of the ATOMS-selected model.
    wealth_baseline:
        Wealth path of the comparison baseline.

    Returns
    -------
    float
        W_T^{ATOMS} / W_T^{baseline} − 1, where W_T is the final entry of
        each series.

    Notes
    -----
    # AMBIGUITY NOTE: Wealth uses raw returns (not excess returns) for
    # economic interpretation.
    """
    w_atoms = float(wealth_atoms.iloc[-1])
    w_baseline = float(wealth_baseline.iloc[-1])
    return w_atoms / w_baseline - 1.0


def compute_industry_wealth_paths(
    daily_forecasts: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """Compute per-industry wealth paths from daily forecasts and returns.

    For each industry, derives sign-based signals from the forecasts and
    accumulates wealth via :func:`compute_wealth_path`.

    Parameters
    ----------
    daily_forecasts:
        DataFrame of daily return forecasts; columns are industry names.
    daily_returns:
        DataFrame of daily realised returns; must share the same columns
        and index alignment as *daily_forecasts*.

    Returns
    -------
    dict
        ``{industry_name: wealth_series}`` — one entry per column.
    """
    wealth_paths: Dict[str, pd.Series] = {}
    for industry in daily_forecasts.columns:
        signals = compute_trading_signals(daily_forecasts[industry])
        wealth_paths[industry] = compute_wealth_path(
            signals, daily_returns[industry]
        )
    return wealth_paths
