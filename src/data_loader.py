"""
data_loader.py
==============
Data loading and factor construction for the nonstationarity-complexity
tradeoff experiments using 17 Kenneth French industry portfolios.

Public API
----------
download_kf_industries          – KF 17 industry portfolio daily returns
download_ff3_daily              – Fama-French 3-factor daily data
generate_chen_factors           – Synthetic Chen et al. (2024) monthly factors
generate_gu_characteristic_factors – Synthetic Gu et al. (2020) char portfolios
build_predictor_panel           – Assemble 129-feature X and 17-target y
load_full_panel                 – Orchestrate all downloads; cache to Parquet
get_recession_masks             – Boolean masks for NBER recession windows
"""

import os
import warnings

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _synthetic_industry_returns(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Return correlated-normal synthetic industry returns when download fails."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    industry_names = [
        'Food ', 'Mines', 'Oil  ', 'Clths', 'Durbl',
        'Chems', 'Cnsum', 'Cnstr', 'Steel', 'FabPr',
        'Machn', 'Cars ', 'Trans', 'Utils', 'Rtail',
        'Finan', 'Other',
    ]
    market = rng.normal(0.0004, 0.010, n)
    betas = rng.uniform(0.6, 1.4, len(industry_names))
    idio = rng.normal(0.0, 0.008, (n, len(industry_names)))
    data = market[:, None] * betas[None, :] + idio
    return pd.DataFrame(data, index=dates, columns=industry_names)


def _synthetic_ff3(start: str, end: str, seed: int = 1) -> pd.DataFrame:
    """Return synthetic FF3 factors when download fails."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    data = {
        'Mkt-RF': rng.normal(0.0004, 0.010, n),
        'SMB':    rng.normal(0.0001, 0.005, n),
        'HML':    rng.normal(0.0001, 0.005, n),
        'RF':     np.full(n, 5e-5),
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Public download functions
# ---------------------------------------------------------------------------

def download_kf_industries(
    start: str = '1987-09-01',
    end: str = '2016-11-30',
) -> pd.DataFrame:
    """
    Download the 17 Kenneth French Industry Portfolios daily returns.

    Uses ``pandas_datareader`` to fetch ``'17_Industry_Portfolios_daily'``
    from the Fama-French data library and returns decimal returns filtered to
    the requested date window.

    Missing observations are forward-filled first; any rows still containing
    NaN are dropped.  If the download fails entirely (network unavailable,
    server outage, etc.) a warning is emitted and synthetic data with the
    same statistical characteristics are returned as a fallback.

    Parameters
    ----------
    start : str
        Panel start date, inclusive, in ``'YYYY-MM-DD'`` format.
    end : str
        Panel end date, inclusive, in ``'YYYY-MM-DD'`` format.

    Returns
    -------
    pd.DataFrame
        Daily returns with a ``DatetimeIndex`` and 17 industry columns.
        Values are decimal returns (e.g., 0.01 for 1 %).
    """
    try:
        raw = web.DataReader(
            '17_Industry_Portfolios_daily', 'famafrench',
            start=start, end=end,
        )
        df = raw[0].copy()
        df = df / 100.0
        # pandas_datareader may return a PeriodIndex; convert to Timestamp first
        if hasattr(df.index, 'to_timestamp'):
            df.index = df.index.to_timestamp()
        else:
            df.index = pd.to_datetime(df.index)
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        df = df.ffill().dropna()
        return df
    except Exception as exc:
        warnings.warn(
            f"Could not download 17_Industry_Portfolios_daily ({exc}). "
            "Falling back to synthetic industry returns.",
            stacklevel=2,
        )
        return _synthetic_industry_returns(start, end)


def download_ff3_daily(
    start: str = '1987-09-01',
    end: str = '2016-11-30',
) -> pd.DataFrame:
    """
    Download the Fama-French 3-Factor daily data.

    Uses ``pandas_datareader`` to fetch
    ``'F-F_Research_Data_Factors_daily'`` from the Fama-French data library.
    Returns decimal returns for the market excess return (Mkt-RF), size
    (SMB), and value (HML) factors plus the risk-free rate (RF).

    Parameters
    ----------
    start : str
        Panel start date, inclusive, in ``'YYYY-MM-DD'`` format.
    end : str
        Panel end date, inclusive, in ``'YYYY-MM-DD'`` format.

    Returns
    -------
    pd.DataFrame
        Daily factor data with columns ``['Mkt-RF', 'SMB', 'HML', 'RF']``
        and a ``DatetimeIndex``.  Values are decimal returns.
    """
    try:
        raw = web.DataReader(
            'F-F_Research_Data_Factors_daily', 'famafrench',
            start=start, end=end,
        )
        df = raw[0].copy()
        df = df / 100.0
        # pandas_datareader may return a PeriodIndex; convert to Timestamp first
        if hasattr(df.index, 'to_timestamp'):
            df.index = df.index.to_timestamp()
        else:
            df.index = pd.to_datetime(df.index)
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        df = df.ffill().dropna()
        return df[['Mkt-RF', 'SMB', 'HML', 'RF']]
    except Exception as exc:
        warnings.warn(
            f"Could not download F-F_Research_Data_Factors_daily ({exc}). "
            "Falling back to synthetic FF3 factors.",
            stacklevel=2,
        )
        return _synthetic_ff3(start, end)


# ---------------------------------------------------------------------------
# Synthetic factor generators
# ---------------------------------------------------------------------------

def generate_chen_factors(
    dates: pd.DatetimeIndex,
    n_factors: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic Chen et al. (2024) monthly factors mapped to daily frequency.

    NOTE: Chen et al. (2024) factors not publicly available. Synthetic AR(1)
    proxies used.

    Creates 15 monthly factors with an AR(1) structure representative of the
    Chen et al. (2024) SDF and latent-factor decomposition:

    * **1 SDF factor** – high persistence (phi=0.85, sigma=0.02)
    * **10 beta-sorted decile factors** – moderate persistence
      (phi=0.70, sigma=0.015), with slightly different unconditional means
      across deciles to mimic the spread in expected returns.
    * **4 macro hidden-state factors** – very persistent
      (phi=0.90, sigma=0.01), representing slow-moving business-cycle states.

    Monthly realisations are broadcast to every trading day within the same
    calendar month, so all days sharing a month carry the same factor value.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading-day index; typically the aligned index of the predictor panel.
    n_factors : int, optional
        Total number of factors to generate (default 15; must equal 15 for the
        standard experimental setup).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shape ``(len(dates), 15)`` with the same ``DatetimeIndex`` as
        ``dates``.  Columns are
        ``'chen_sdf'``,
        ``'chen_decile_01'`` … ``'chen_decile_10'``,
        ``'chen_macro_1'`` … ``'chen_macro_4'``.
    """
    rng = np.random.default_rng(seed)

    # Build a monthly period range that covers the full date span
    monthly_start = dates.min().to_period('M')
    monthly_end   = dates.max().to_period('M')
    monthly_index = pd.period_range(monthly_start, monthly_end, freq='M')
    T_m = len(monthly_index)

    def _ar1(phi: float, sigma: float, T: int, mean: float = 0.0) -> np.ndarray:
        """Simulate one AR(1) series of length T."""
        x = np.empty(T)
        x[0] = rng.normal(mean, sigma / np.sqrt(max(1 - phi ** 2, 1e-8)))
        noise = rng.standard_normal(T - 1)
        for t in range(1, T):
            x[t] = mean + phi * (x[t - 1] - mean) + sigma * noise[t - 1]
        return x

    factor_cols: Dict[str, np.ndarray] = {}

    # 1 SDF factor
    factor_cols['chen_sdf'] = _ar1(phi=0.85, sigma=0.02, T=T_m)

    # 10 beta-sorted decile factors
    for i in range(1, 11):
        mean_shift = (i - 5.5) * 0.001   # slight spread across deciles
        factor_cols[f'chen_decile_{i:02d}'] = _ar1(
            phi=0.70, sigma=0.015, T=T_m, mean=mean_shift,
        )

    # 4 macro hidden-state factors
    for i in range(1, 5):
        factor_cols[f'chen_macro_{i}'] = _ar1(phi=0.90, sigma=0.01, T=T_m)

    monthly_df = pd.DataFrame(factor_cols, index=monthly_index)

    # Broadcast each month's value to every trading day in that month
    day_periods = dates.to_period('M')
    daily_values = monthly_df.reindex(day_periods).values  # shape (len(dates), 15)
    return pd.DataFrame(daily_values, index=dates, columns=monthly_df.columns)


def generate_gu_characteristic_factors(
    dates: pd.DatetimeIndex,
    n_chars: int = 94,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily long-short characteristic portfolio returns.

    NOTE: Gu et al. (2020) characteristics require CRSP/WRDS access. Synthetic
    factor-model proxies used.

    Produces 94 daily long-short portfolio returns inspired by the
    characteristic-sorted portfolios of Gu, Kelly & Xiu (2020).  Returns are
    generated via a latent factor model::

        r_{i,t} = L_i @ F_t + eps_{i,t}

    where **F_t** is a 5-dimensional vector of latent factors following
    mean-reverting AR(1) processes (phi=0.05, sigma=0.01) and **L_i** are
    sparse factor loadings (each characteristic loads on 1–2 latent factors
    with random signs and magnitudes).  Idiosyncratic noise
    ``eps_{i,t} ~ N(0, 0.005^2)``.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading-day index aligned to the predictor panel.
    n_chars : int, optional
        Number of characteristic long-short portfolios (default 94).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shape ``(len(dates), 94)`` with the same ``DatetimeIndex`` as
        ``dates``.  Columns are named ``'char_ls_01'`` through
        ``'char_ls_94'``.
    """
    rng = np.random.default_rng(seed)
    T = len(dates)
    n_latent = 5
    phi_f, sigma_f = 0.05, 0.01

    # Latent factor matrix F, shape (T, n_latent)
    F = np.zeros((T, n_latent))
    F[0] = rng.normal(0.0, sigma_f / np.sqrt(max(1 - phi_f ** 2, 1e-8)), n_latent)
    eps_f = rng.normal(0.0, sigma_f, (T - 1, n_latent))
    for t in range(1, T):
        F[t] = phi_f * F[t - 1] + eps_f[t - 1]

    # Sparse loadings matrix L, shape (n_chars, n_latent)
    L = np.zeros((n_chars, n_latent))
    for i in range(n_chars):
        primary = int(rng.integers(0, n_latent))
        L[i, primary] = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
        if rng.random() < 0.4:          # 40 % chance of a secondary loading
            secondary = int(rng.integers(0, n_latent))
            while secondary == primary:
                secondary = int(rng.integers(0, n_latent))
            L[i, secondary] = rng.uniform(0.1, 0.5) * rng.choice([-1.0, 1.0])

    idio = rng.normal(0.0, 0.005, (T, n_chars))
    returns = F @ L.T + idio          # shape (T, n_chars)

    cols = [f'char_ls_{i:02d}' for i in range(1, n_chars + 1)]
    return pd.DataFrame(returns, index=dates, columns=cols)


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

def build_predictor_panel(
    industry_returns: pd.DataFrame,
    ff3: pd.DataFrame,
    chen_factors: pd.DataFrame,
    gu_factors: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble the 129-feature predictor matrix X and 17-target matrix y.

    Predictor block layout (no look-ahead bias):

    =========  ======  ================================================
    Columns    Count   Description
    =========  ======  ================================================
    0 – 16        17   Lagged industry returns at *t − 1*
    17 – 19        3   Fama-French factors at *t* (Mkt-RF, SMB, HML)
    20 – 34       15   Chen et al. monthly factors (mapped to daily)
    35 – 128      94   Gu et al. characteristic long-short portfolios
    =========  ======  ================================================

    The method adds two auxiliary grouping columns (``'month'``, ``'year'``)
    to X after validating the 129-feature count, yielding a final shape of
    (N, 131).

    Parameters
    ----------
    industry_returns : pd.DataFrame
        Daily industry returns, shape (T, 17).
    ff3 : pd.DataFrame
        Daily FF3 factors; must contain columns ``'Mkt-RF'``, ``'SMB'``,
        ``'HML'``.
    chen_factors : pd.DataFrame
        Daily (monthly-broadcast) Chen factors, shape (T, 15).
    gu_factors : pd.DataFrame
        Daily characteristic long-short returns, shape (T, 94).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        **X** – predictor panel, shape (N, 131): 129 features plus
        ``'month'`` and ``'year'`` grouping columns.

        **y** – target panel, shape (N, 17): industry returns at time *t*.

    Raises
    ------
    ValueError
        If the assembled feature matrix does not have exactly 129 predictors
        before the grouping columns are appended.
    """
    # Step 1: align to the intersection of all trading-day indices
    common_idx = (
        industry_returns.index
        .intersection(ff3.index)
        .intersection(chen_factors.index)
        .intersection(gu_factors.index)
    )
    ind           = industry_returns.loc[common_idx]
    ff3_aligned   = ff3.loc[common_idx]
    chen_aligned  = chen_factors.loc[common_idx]
    gu_aligned    = gu_factors.loc[common_idx]

    # Step 2: lagged industry returns (t-1) — 17 features, no look-ahead
    lagged_ind = ind.shift(1)
    lagged_ind.columns = [f'lag_{c}' for c in ind.columns]

    # Step 3: FF3 factors at t — 3 features (computed after market close)
    ff3_feats = ff3_aligned[['Mkt-RF', 'SMB', 'HML']]

    # Steps 4 & 5: Chen (15) and Gu (94) factors already aligned
    # Step 6: concatenate horizontally → 17 + 3 + 15 + 94 = 129 columns
    X_raw = pd.concat([lagged_ind, ff3_feats, chen_aligned, gu_aligned], axis=1)

    # Step 7: target matrix
    y = ind.copy()

    # Step 8: drop any row with NaN in X or y (first row has NaN due to lag)
    valid = X_raw.notna().all(axis=1) & y.notna().all(axis=1)
    X_raw = X_raw.loc[valid]
    y     = y.loc[valid]

    # Step 9: guard on feature count
    if X_raw.shape[1] != 129:
        raise ValueError(
            f"Expected 129 predictor columns, got {X_raw.shape[1]}. "
            "Verify that input DataFrames have the correct column counts: "
            "industry(17), ff3(3), chen(15), gu(94)."
        )

    # Append grouping helpers (month, year) for rolling/expanding CV splits
    X = X_raw.copy()
    X['month'] = X.index.month
    X['year']  = X.index.year

    return X, y


# ---------------------------------------------------------------------------
# Master loader with Parquet cache
# ---------------------------------------------------------------------------

def load_full_panel(
    start: str = '1987-09-01',
    end: str = '2016-11-30',
    cache_dir: str = 'data/cache',
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate all downloads / synthesis and return the aligned (X, y) panel.

    Checks for cached Parquet files at ``{cache_dir}/X_panel.parquet`` and
    ``{cache_dir}/y_panel.parquet``.  If both exist and ``force_refresh`` is
    False the cached panels are returned directly, avoiding redundant network
    calls.  Otherwise the full pipeline runs:

    1. :func:`download_kf_industries` — 17 industry daily returns
    2. :func:`download_ff3_daily` — Fama-French 3-factor daily returns
    3. :func:`generate_chen_factors` — synthetic Chen (2024) monthly factors
    4. :func:`generate_gu_characteristic_factors` — synthetic Gu (2020) chars
    5. :func:`build_predictor_panel` — align and assemble (X, y)
    6. Save Parquet cache

    Parameters
    ----------
    start : str
        Panel start date (inclusive), ``'YYYY-MM-DD'``.
    end : str
        Panel end date (inclusive), ``'YYYY-MM-DD'``.
    cache_dir : str
        Directory in which to read/write Parquet cache files.  Created
        automatically if it does not exist.
    force_refresh : bool
        If True, ignore existing cache and re-download/regenerate everything.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        **X** – predictor panel, shape (N, 131).
        **y** – target panel, shape (N, 17).
    """
    os.makedirs(cache_dir, exist_ok=True)
    x_path = os.path.join(cache_dir, 'X_panel.parquet')
    y_path = os.path.join(cache_dir, 'y_panel.parquet')

    if not force_refresh and os.path.exists(x_path) and os.path.exists(y_path):
        print("Loading predictor panel from cache …")
        X = pd.read_parquet(x_path)
        y = pd.read_parquet(y_path)
        print(f"  X: {X.shape}  |  y: {y.shape}")
        print(f"  Date range: {X.index[0].date()} → {X.index[-1].date()}")
        return X, y

    print("Downloading / generating factor data …")
    industry_returns = download_kf_industries(start=start, end=end)
    ff3              = download_ff3_daily(start=start, end=end)

    # Generate synthetic factors on the industry return date grid
    chen = generate_chen_factors(dates=industry_returns.index)
    gu   = generate_gu_characteristic_factors(dates=industry_returns.index)

    print("Assembling predictor panel …")
    X, y = build_predictor_panel(industry_returns, ff3, chen, gu)

    print(f"Saving to {cache_dir} …")
    X.to_parquet(x_path)
    y.to_parquet(y_path)

    print(f"  X: {X.shape}  |  y: {y.shape}")
    print(f"  Date range: {X.index[0].date()} → {X.index[-1].date()}")
    return X, y


# ---------------------------------------------------------------------------
# Recession / evaluation period masks
# ---------------------------------------------------------------------------

def get_recession_masks(dates: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    """
    Return boolean masks for NBER recession and out-of-sample evaluation windows.

    Approximate NBER peak-to-trough dates used:

    ============  ==========================  ================================
    Key           Date range                  Description
    ============  ==========================  ================================
    gulf_war      1990-06-01 – 1990-10-31     Gulf War recession
    dot_com       2001-05-01 – 2001-10-31     Dot-com bust / 9-11 aftermath
    gfc           2007-11-01 – 2009-06-30     Global Financial Crisis
    full_oos      1990-01-01 – 2016-11-30     Full out-of-sample window
    ============  ==========================  ================================

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The trading-day index of the predictor panel (e.g., ``X.index``).

    Returns
    -------
    Dict[str, pd.Series]
        Mapping from period name to a boolean ``pd.Series`` indexed by
        ``dates``.  ``True`` where the date falls inside the named window.
    """
    periods = {
        'gulf_war': ('1990-06-01', '1990-10-31'),
        'dot_com':  ('2001-05-01', '2001-10-31'),
        'gfc':      ('2007-11-01', '2009-06-30'),
        'full_oos': ('1990-01-01', '2016-11-30'),
    }
    return {
        name: pd.Series(
            (dates >= pd.Timestamp(p_start)) & (dates <= pd.Timestamp(p_end)),
            index=dates,
            name=name,
        )
        for name, (p_start, p_end) in periods.items()
    }
