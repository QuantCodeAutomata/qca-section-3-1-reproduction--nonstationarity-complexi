"""
Candidate model definitions and training/validation utilities.

The experiment considers 168 candidate models: the Cartesian product of
28 model/hyperparameter combinations with 6 training-window lengths.

Model/hyperparameter combos
----------------------------
* Ridge   — 5 alpha values
* Lasso   — 5 alpha values
* ElasticNet — 3 alpha × 3 l1_ratio = 9
* RandomForest — 3 n_estimators × 3 max_depth = 9
Total per window: 28

Training windows (months of history)
--------------------------------------
1, 4, 16, 64, 256, None (= all available history)
Total: 6

Combined: 28 × 6 = 168 candidates.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from typing import Any, Dict, List, Optional, Tuple
import warnings


@dataclass
class CandidateModel:
    """Specification for one candidate model in the tournament.

    Parameters
    ----------
    model_class:
        One of ``'Ridge'``, ``'Lasso'``, ``'ElasticNet'``, ``'RF'``.
    hyperparams:
        Keyword arguments forwarded to the sklearn constructor
        (e.g. ``{'alpha': 0.01}`` for Ridge).
    window_months:
        Number of calendar months of history used for training.
        ``None`` means use all available history.
    model_id:
        Unique string identifier derived from model class, hyperparams,
        and window.
    """

    model_class: str
    hyperparams: Dict[str, Any]
    window_months: Optional[int]
    model_id: str

    def __repr__(self) -> str:
        return (
            f"CandidateModel(model_class={self.model_class!r}, "
            f"hyperparams={self.hyperparams}, "
            f"window_months={self.window_months!r})"
        )

    @staticmethod
    def all_candidates() -> "List[CandidateModel]":
        """Return the full set of 168 candidate models.

        Constructs the Cartesian product of 28 model/hyperparam combinations
        with 6 training-window sizes.

        Returns
        -------
        list of CandidateModel
            All 168 candidates in a deterministic order:
            Ridge × 5 alphas, Lasso × 5 alphas, ElasticNet × 9 combos,
            RF × 9 combos — each repeated for the 6 window lengths.
        """
        candidates: List[CandidateModel] = []
        windows: List[Optional[int]] = [1, 4, 16, 64, 256, None]

        # ── Ridge (5 alpha values) ────────────────────────────────────────────
        ridge_alphas = [1e-3, 10 ** -1.5, 1.0, 10 ** 1.5, 1e3]
        for alpha in ridge_alphas:
            for w in windows:
                w_tag = "all" if w is None else str(w)
                mid = f"Ridge_alpha{alpha:.4g}_w{w_tag}"
                candidates.append(
                    CandidateModel(
                        model_class="Ridge",
                        hyperparams={"alpha": alpha},
                        window_months=w,
                        model_id=mid,
                    )
                )

        # ── Lasso (5 alpha values) ────────────────────────────────────────────
        lasso_alphas = [1e-5, 10 ** -3.5, 1e-2, 10 ** -0.5, 10.0]
        for alpha in lasso_alphas:
            for w in windows:
                w_tag = "all" if w is None else str(w)
                mid = f"Lasso_alpha{alpha:.4g}_w{w_tag}"
                candidates.append(
                    CandidateModel(
                        model_class="Lasso",
                        hyperparams={"alpha": alpha},
                        window_months=w,
                        model_id=mid,
                    )
                )

        # ── ElasticNet (3 alpha × 3 l1_ratio = 9) ────────────────────────────
        en_alphas = [1e-3, 1.0, 1e3]
        en_l1_ratios = [0.01, 0.05, 0.1]
        for alpha in en_alphas:
            for l1r in en_l1_ratios:
                for w in windows:
                    w_tag = "all" if w is None else str(w)
                    mid = f"ElasticNet_alpha{alpha:.4g}_l1r{l1r:.4g}_w{w_tag}"
                    candidates.append(
                        CandidateModel(
                            model_class="ElasticNet",
                            hyperparams={"alpha": alpha, "l1_ratio": l1r},
                            window_months=w,
                            model_id=mid,
                        )
                    )

        # ── Random Forest (3 n_estimators × 3 max_depth = 9) ─────────────────
        rf_n_estimators = [10, 100, 200]
        rf_max_depths = [3, 5, 10]
        for ne in rf_n_estimators:
            for md in rf_max_depths:
                for w in windows:
                    w_tag = "all" if w is None else str(w)
                    mid = f"RF_ne{ne}_md{md}_w{w_tag}"
                    candidates.append(
                        CandidateModel(
                            model_class="RF",
                            hyperparams={"n_estimators": ne, "max_depth": md},
                            window_months=w,
                            model_id=mid,
                        )
                    )

        return candidates


def get_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    t_month: pd.Period,
    window_months: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract training data for month t with no lookahead.

    Collects all observations whose calendar month is strictly before
    *t_month*, then restricts to the last *window_months* distinct months
    when a finite window is requested.

    Parameters
    ----------
    X:
        Feature matrix with a DatetimeIndex or PeriodIndex.  May contain
        multiple rows per calendar month (e.g. daily observations).
    y:
        Target series aligned with X.
    t_month:
        The OOS month being forecast; no data from this month or later
        may be used.
    window_months:
        Number of calendar months of look-back history to retain.
        ``None`` means keep all available history before *t_month*.

    Returns
    -------
    X_train : np.ndarray, shape (n_obs, n_features)
    y_train : np.ndarray, shape (n_obs,)
        Only rows from months strictly before *t_month*.
    """
    # Map every row to its monthly period for boundary comparisons
    if isinstance(X.index, pd.PeriodIndex):
        freq = getattr(X.index, "freqstr", "")
        if freq.startswith("M"):
            row_months: pd.PeriodIndex = X.index
        else:
            row_months = X.index.to_timestamp().to_period("M")
    else:
        row_months = X.index.to_period("M")

    # Boolean mask: strictly before t_month (no lookahead).
    # np.asarray handles both pandas Index comparisons (which may return a
    # numpy array directly) and any object that has a .values attribute.
    before_mask: np.ndarray = np.asarray(row_months < t_month)

    X_vals: np.ndarray = X.values[before_mask]
    y_vals: np.ndarray = y.values[before_mask]
    months_hist: pd.PeriodIndex = row_months[before_mask]

    if window_months is not None and len(X_vals) > 0:
        unique_months: List[pd.Period] = sorted(months_hist.unique())
        if len(unique_months) > window_months:
            cutoff: pd.Period = unique_months[-window_months]
            window_mask: np.ndarray = np.asarray(months_hist >= cutoff)
            X_vals = X_vals[window_mask]
            y_vals = y_vals[window_mask]

    return X_vals, y_vals


def subsample_month(
    X_month: np.ndarray,
    y_month: np.ndarray,
    frac: float = 0.8,
    seed: int = 0,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Randomly partition one month's observations into train and validation.

    Uses a seeded RNG so results are reproducible.  The split is random
    (not sequential) within the month, giving an 80/20 split by default.

    Parameters
    ----------
    X_month:
        Feature matrix for a single calendar month, shape (n, p).
    y_month:
        Target array for the same month, shape (n,).
    frac:
        Fraction of observations allocated to the training subset (default
        0.8, corresponding to 4/5 train, 1/5 validation as in the paper).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (X_tr, y_tr) : train subset
    (X_va, y_va) : validation subset
    """
    rng = np.random.RandomState(seed)
    n = len(y_month)
    n_train = int(round(n * frac))
    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return (X_month[train_idx], y_month[train_idx]), (X_month[val_idx], y_month[val_idx])


def fit_candidate(
    candidate: CandidateModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 0,
) -> Any:
    """Fit a sklearn estimator according to the candidate's specification.

    The intercept is *not* penalised for linear models under sklearn's
    default ``fit_intercept=True`` behaviour.

    Parameters
    ----------
    candidate:
        Specification object carrying ``model_class`` and ``hyperparams``.
    X_train:
        Training features, shape (n, p).
    y_train:
        Training targets, shape (n,).
    random_state:
        Seed passed to stochastic estimators (unused for deterministic ones).

    Returns
    -------
    fitted sklearn estimator
        A model that exposes a ``predict(X)`` method.

    Notes
    -----
    # AMBIGUITY NOTE: sklearn does not penalize the intercept under
    # fit_intercept=True; following sklearn behaviour as primary implementation.
    """
    hp = candidate.hyperparams

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if candidate.model_class == "Ridge":
            model = Ridge(alpha=hp["alpha"], fit_intercept=True)

        elif candidate.model_class == "Lasso":
            model = Lasso(alpha=hp["alpha"], fit_intercept=True, max_iter=10000)

        elif candidate.model_class == "ElasticNet":
            model = ElasticNet(
                alpha=hp["alpha"],
                l1_ratio=hp["l1_ratio"],
                fit_intercept=True,
                max_iter=10000,
            )

        elif candidate.model_class == "RF":
            model = RandomForestRegressor(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                random_state=0,
                n_jobs=-1,   # use all available cores to speed up tree building
            )

        else:
            raise ValueError(f"Unknown model_class: {candidate.model_class!r}")

        model.fit(X_train, y_train)

    return model


def get_monthly_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    t_month: pd.Period,
    window_months: Optional[int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return per-month training data as a list of (X_m, y_m) numpy-array tuples.

    Iterates over all calendar months strictly before *t_month* and returns
    one ``(X_m, y_m)`` pair per month in chronological order.  The optional
    *window_months* parameter limits the look-back to the most recent months.

    This helper enables per-month 4/5 subsampling as described in Section 3.1:
    each month is subsampled independently before pooling training observations.

    Parameters
    ----------
    X:
        Feature panel with a DatetimeIndex.
    y:
        Target series aligned with X.
    t_month:
        OOS month being forecast; no data from this month or later is used.
    window_months:
        Number of most-recent calendar months to include.  ``None`` means
        keep all available history before *t_month*.

    Returns
    -------
    list of (X_m, y_m) tuples
        Each element is a month's worth of observations as numpy arrays.
        The list is in ascending chronological order.
    """
    if isinstance(X.index, pd.PeriodIndex):
        row_months = X.index.to_timestamp().to_period("M")
    else:
        row_months = X.index.to_period("M")

    before_mask = np.asarray(row_months < t_month)
    months_hist = row_months[before_mask]
    X_hist = X.values[before_mask]
    y_hist = y.values[before_mask]

    unique_months: List[pd.Period] = sorted(months_hist.unique())
    if window_months is not None and len(unique_months) > window_months:
        unique_months = unique_months[-window_months:]

    monthly_data: List[Tuple[np.ndarray, np.ndarray]] = []
    for m in unique_months:
        m_mask = np.asarray(months_hist == m)
        monthly_data.append((X_hist[m_mask], y_hist[m_mask]))

    return monthly_data


def compute_validation_errors(
    fitted_model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> np.ndarray:
    """Compute per-observation squared prediction errors on a validation set.

    Parameters
    ----------
    fitted_model:
        A fitted sklearn estimator with a ``predict`` method.
    X_val:
        Validation features, shape (n, p).
    y_val:
        Validation targets, shape (n,).

    Returns
    -------
    np.ndarray, shape (n,)
        Squared error for each validation observation:
        ``(y_val - fitted_model.predict(X_val))**2``.
    """
    y_hat = fitted_model.predict(X_val)
    return (y_val - y_hat) ** 2
