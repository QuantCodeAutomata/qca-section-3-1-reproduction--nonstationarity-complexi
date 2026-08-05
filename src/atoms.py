"""
ATOMS — Adaptive Tournament Model Selection.

Implements the ATOMS algorithm from the paper on nonstationarity-complexity
tradeoffs.  The core idea is to run pairwise model comparisons using an
*adaptive* validation horizon that automatically adjusts for structural breaks.

Paper hyperparameters
---------------------
* delta_prime = 0.1   (confidence level Δ')
* M_squared   = 5e-4  (bound on the range of individual losses, M²)

Notation (matching the paper)
------------------------------
* u_{j,i}  = sq_error_f1_{j,i} − sq_error_f2_{j,i}   (per observation)
* n_{t,l}  = total observations in the last l months (horizon l)
* Δ̂(t,l)  = mean of u values over the horizon l
* ψ̂(t,l)  = empirical width of the confidence interval
* φ̂(t,l)  = nonstationarity-detection statistic
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Paper constants ────────────────────────────────────────────────────────────

DELTA_PRIME: float = 0.1   # ATOMS hyperparameter delta'
M_SQUARED: float = 5e-4    # ATOMS hyperparameter M^2


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class ValidationHistory:
    """Per-month validation statistics for a single candidate model.

    Attributes
    ----------
    val_errors_by_month:
        Maps month_index (0-based, ascending in time) to a 1-D array of
        squared prediction errors for the validation subsample of that month.
    predictions_by_month:
        Maps month_index to a 1-D array of raw predictions for the
        validation subsample (stored for diagnostics / post-hoc analysis).
    """

    val_errors_by_month: Dict[int, np.ndarray] = field(default_factory=dict)
    predictions_by_month: Dict[int, np.ndarray] = field(default_factory=dict)


# ── Primitive statistics ───────────────────────────────────────────────────────

def compute_delta_hat(u_values: np.ndarray) -> float:
    """Compute the empirical loss difference Δ̂(t, l).

    This is the average of ``u_{j,i} = sq_err_f1_{j,i} − sq_err_f2_{j,i}``
    over all validation observations in the chosen horizon.

    Parameters
    ----------
    u_values:
        1-D array of per-observation loss differences (f1 minus f2 squared
        errors), aggregated over the l most recent months.

    Returns
    -------
    float
        Mean of *u_values*.  Negative values favour f1; positive favour f2.
    """
    return float(np.mean(u_values))


def compute_psi_hat(
    u_values: np.ndarray,
    delta_prime: float = DELTA_PRIME,
    m_squared: float = M_SQUARED,
) -> float:
    """Compute the confidence-interval half-width ψ̂(t, l).

    Piecewise formula from the paper (ATOMS equation):

    * n = 1  →  ψ̂ = 8 · M²
    * n > 1  →  ψ̂ = v̂ · √(2 · ln(2/Δ') / n) + 64 · M² · ln(2/Δ') / (3(n−1))

    where v̂ is the *sample* standard deviation (ddof = 1) of *u_values*.

    # Custom — Context7 found no library equivalent (paper ATOMS Eq.)

    Parameters
    ----------
    u_values:
        1-D array of per-observation loss differences for horizon l.
    delta_prime:
        Confidence parameter Δ' (default 0.1 from the paper).
    m_squared:
        Bound parameter M² (default 5e-4 from the paper).

    Returns
    -------
    float
        ψ̂(t, l) — the confidence-interval half-width for this horizon.
    """
    n = len(u_values)
    if n == 1:
        return 8.0 * m_squared

    v_hat = float(np.std(u_values, ddof=1))
    log_term = np.log(2.0 / delta_prime)
    psi_hat = (
        v_hat * np.sqrt(2.0 * log_term / n)
        + 64.0 * m_squared * log_term / (3.0 * (n - 1))
    )
    return float(psi_hat)


def compute_phi_hat(
    delta_hats: List[float],
    psi_hats: List[float],
) -> List[float]:
    """Compute the nonstationarity-detection statistic φ̂(t, l) for each horizon.

    For each horizon l (1-indexed), the statistic measures the maximum
    discrepancy between the loss-difference estimate at horizon l and at any
    shorter horizon i, net of their combined confidence widths:

        φ̂(t, l) = max_{i=1,...,l}  (|Δ̂(t,l) − Δ̂(t,i)| − ψ̂(t,l) − ψ̂(t,i))₊

    where (·)₊ = max(·, 0).

    Parameters
    ----------
    delta_hats:
        List of length L: ``delta_hats[l-1]`` = Δ̂(t, l).
    psi_hats:
        List of length L: ``psi_hats[l-1]`` = ψ̂(t, l).

    Returns
    -------
    list of float, length L
        ``phi_hats[l-1]`` = φ̂(t, l) for l = 1, …, L.
    """
    L = len(delta_hats)
    phi_hats: List[float] = []
    for l in range(1, L + 1):           # l is 1-indexed
        max_disc = 0.0
        for i in range(1, l + 1):       # i ∈ {1, …, l}
            disc = (
                abs(delta_hats[l - 1] - delta_hats[i - 1])
                - psi_hats[l - 1]
                - psi_hats[i - 1]
            )
            if disc > max_disc:
                max_disc = disc
        phi_hats.append(max_disc)
    return phi_hats


# ── Horizon selection ──────────────────────────────────────────────────────────

def select_validation_horizon(
    u_values_by_horizon: List[np.ndarray],
    delta_prime: float = DELTA_PRIME,
    m_squared: float = M_SQUARED,
) -> Tuple[int, float]:
    """Select the adaptive validation horizon l̂ via the ATOMS criterion.

    For each candidate horizon l the criterion trades off the confidence
    interval width ψ̂(t,l) against the nonstationarity penalty φ̂(t,l):

        l̂ = argmin_{l=1,...,L} [φ̂(t,l) + ψ̂(t,l)]

    Parameters
    ----------
    u_values_by_horizon:
        Length-L list where ``u_values_by_horizon[l-1]`` is the array of
        per-observation loss differences accumulated over the last *l* months.
        Arrays are strictly nested: ``u_values_by_horizon[l-1]`` is a
        superset of ``u_values_by_horizon[l-2]``.
    delta_prime:
        Confidence parameter Δ'.
    m_squared:
        Range-bound parameter M².

    Returns
    -------
    l_hat : int
        Selected horizon (1-indexed).
    delta_hat_at_l_hat : float
        Δ̂(t, l̂) — the empirical loss difference at the selected horizon.
    """
    L = len(u_values_by_horizon)
    delta_hats: List[float] = []
    psi_hats: List[float] = []

    for l in range(1, L + 1):
        u = u_values_by_horizon[l - 1]
        delta_hats.append(compute_delta_hat(u))
        psi_hats.append(compute_psi_hat(u, delta_prime=delta_prime, m_squared=m_squared))

    phi_hats = compute_phi_hat(delta_hats, psi_hats)

    # Criterion: argmin over l of (phi_hat[l] + psi_hat[l])
    criterion = [phi_hats[l] + psi_hats[l] for l in range(L)]
    l_hat_idx = int(np.argmin(criterion))   # 0-indexed
    l_hat = l_hat_idx + 1                   # 1-indexed

    return l_hat, delta_hats[l_hat_idx]


# ── Pairwise comparison ────────────────────────────────────────────────────────

def atoms_pairwise_compare(
    val_errors_f1_by_month: Dict[int, np.ndarray],
    val_errors_f2_by_month: Dict[int, np.ndarray],
    available_horizons: int,
) -> Tuple[int, dict]:
    """Run the ATOMS pairwise comparison between two candidate models.

    Constructs the per-observation loss differences u_{j,i} for each horizon
    l = 1, …, available_horizons and delegates to
    :func:`select_validation_horizon`.

    # Custom — Context7 found no library equivalent (ATOMS algorithm)

    Parameters
    ----------
    val_errors_f1_by_month:
        ``{month_idx: np.ndarray}`` of squared errors for model f1.
        Month indices run 0, 1, …, T−1 with T−1 being the most recent.
    val_errors_f2_by_month:
        Same structure for model f2.
    available_horizons:
        Number of past months with stored validation errors; the most recent
        month has index ``available_horizons − 1``.

    Returns
    -------
    winner : int
        1 if f1 wins (Δ̂ ≤ 0), 2 if f2 wins (Δ̂ > 0).
    diagnostics : dict
        ``{'l_hat': int, 'delta_hat': float}`` for post-hoc inspection.
    """
    u_values_by_horizon: List[np.ndarray] = []

    for l in range(1, available_horizons + 1):
        # Months used for horizon l: from (T - l) to (T - 1) inclusive
        start_month = available_horizons - l
        pieces: List[np.ndarray] = []
        for j in range(start_month, available_horizons):
            if j in val_errors_f1_by_month and j in val_errors_f2_by_month:
                u_j = val_errors_f1_by_month[j] - val_errors_f2_by_month[j]
                pieces.append(u_j)
        if pieces:
            u_values_by_horizon.append(np.concatenate(pieces))
        else:
            # No shared observations for this horizon — pad with a zero scalar
            u_values_by_horizon.append(np.array([0.0]))

    l_hat, delta_hat = select_validation_horizon(u_values_by_horizon)
    winner = 1 if delta_hat <= 0.0 else 2
    return winner, {"l_hat": l_hat, "delta_hat": delta_hat}


# ── Tournament elimination ─────────────────────────────────────────────────────

def run_atoms_tournament(
    candidates: List[Any],
    val_errors_by_candidate_month: Dict[int, Dict[int, np.ndarray]],
    available_horizons: int,
    pivot_seed: int = 0,
) -> Tuple[int, List[dict]]:
    """Run the full ATOMS elimination tournament.

    Repeatedly selects a random pivot from the surviving set, compares it
    against all other survivors, and replaces the set with the challengers
    that beat the pivot.  The pivot wins the tournament if no challenger
    defeats it.

    # Custom — Context7 found no library equivalent (ATOMS algorithm)

    Parameters
    ----------
    candidates:
        List of candidate indices (ints) in the initial pool.
    val_errors_by_candidate_month:
        Nested dict ``{cand_idx: {month_idx: np.ndarray of sq errors}}``.
    available_horizons:
        Number of past months available for the ATOMS comparison.
    pivot_seed:
        RNG seed for reproducible pivot selection.

    Returns
    -------
    winner_idx : int
        Index of the tournament winner.
    history : list of dict
        One entry per pairwise comparison, each containing ``'pivot'``,
        ``'challenger'``, ``'winner'``, ``'l_hat'``, and ``'delta_hat'``.
    """
    S: List[int] = list(candidates)
    rng = np.random.RandomState(pivot_seed)
    history: List[dict] = []

    while len(S) > 1:
        pivot_idx: int = int(rng.choice(S))
        challengers_that_beat_pivot: List[int] = []

        for c in S:
            if c == pivot_idx:
                continue
            winner, diag = atoms_pairwise_compare(
                val_errors_by_candidate_month[pivot_idx],
                val_errors_by_candidate_month[c],
                available_horizons,
            )
            if winner == 2:
                # Challenger (f2 = c) beat the pivot (f1 = pivot_idx)
                challengers_that_beat_pivot.append(c)
            history.append(
                {"pivot": pivot_idx, "challenger": c, "winner": winner, **diag}
            )

        if not challengers_that_beat_pivot:
            # Pivot is unbeaten — it wins the tournament
            return pivot_idx, history

        # Reduce surviving set to those that defeated the pivot
        S = challengers_that_beat_pivot

    return S[0], history


# ── Baseline selection methods ─────────────────────────────────────────────────

def select_fixed_val(
    val_errors_by_candidate_month: Dict[int, Dict[int, np.ndarray]],
    available_horizons: int,
    window_months: int,
) -> int:
    """Fixed-val(l) baseline: select the candidate with minimum validation SSE.

    Uses a fixed look-back of ``l = min(window_months, available_horizons)``
    months rather than the adaptive horizon chosen by ATOMS.

    # Custom — Context7 found no library equivalent (ATOMS algorithm)

    Parameters
    ----------
    val_errors_by_candidate_month:
        Nested dict ``{cand_idx: {month_idx: np.ndarray of sq errors}}``.
    available_horizons:
        Total number of past months with stored validation errors.
    window_months:
        Requested look-back length; capped at *available_horizons*.

    Returns
    -------
    int
        Index of the candidate with the lowest total validation SSE over
        the selected window.
    """
    l = min(window_months, available_horizons)
    start_month = available_horizons - l

    best_idx: Optional[int] = None
    best_sse: float = float("inf")

    for cand_idx, errors_by_month in val_errors_by_candidate_month.items():
        total_sse = 0.0
        for j in range(start_month, available_horizons):
            if j in errors_by_month:
                total_sse += float(np.sum(errors_by_month[j]))
        if total_sse < best_sse:
            best_sse = total_sse
            best_idx = cand_idx

    assert best_idx is not None, "No candidates provided."
    return best_idx


def select_fixed_cv(
    candidates_train_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    cv_window: int = 36,
    n_folds: int = 5,
    candidate_specs: Optional[Dict[int, Any]] = None,
) -> int:
    """Fixed-CV baseline: select the candidate with the lowest k-fold CV loss.

    Pools all daily observations from each candidate's training data (assumed
    to already cover the last *cv_window* months) and evaluates each candidate
    via sequential k-fold cross-validation.

    # AMBIGUITY NOTE: Fold construction: sequential split of pooled daily
    # observations from last 36 months.  Model fitting uses ``fit_candidate``
    # when ``candidate_specs`` are provided; Ridge(alpha=1.0) is used as a
    # fallback when specs are absent.

    Parameters
    ----------
    candidates_train_data:
        ``{cand_idx: (X_train, y_train)}`` — numpy arrays for each candidate,
        expected to contain the observations from the last *cv_window* months.
    cv_window:
        Length of the evaluation window in months (default 36, i.e. 3 years).
    n_folds:
        Number of sequential CV folds (default 5).
    candidate_specs:
        Optional ``{cand_idx: CandidateModel}`` mapping.  When supplied the
        actual candidate estimator is fitted in each fold; otherwise
        ``Ridge(alpha=1.0)`` is used for all candidates.

    Returns
    -------
    int
        Index of the candidate with the minimum average CV squared error.
    """
    best_idx: Optional[int] = None
    best_cv_loss: float = float("inf")

    for cand_idx, (X_train, y_train) in candidates_train_data.items():
        n_obs = len(y_train)
        if n_obs < n_folds:
            # Too few observations — skip; assign a large loss
            avg_cv_loss = float("inf")
        else:
            fold_losses: List[float] = []
            fold_boundaries = [
                (k * n_obs // n_folds, (k + 1) * n_obs // n_folds)
                for k in range(n_folds)
            ]
            for k, (val_start, val_end) in enumerate(fold_boundaries):
                # Build train indices (all folds except k)
                train_mask = np.ones(n_obs, dtype=bool)
                train_mask[val_start:val_end] = False

                X_tr = X_train[train_mask]
                y_tr = y_train[train_mask]
                X_va = X_train[val_start:val_end]
                y_va = y_train[val_start:val_end]

                if len(y_tr) == 0 or len(y_va) == 0:
                    continue

                # Fit the appropriate model
                if candidate_specs is not None and cand_idx in candidate_specs:
                    from src.models import fit_candidate as _fit  # noqa: PLC0415
                    fitted = _fit(candidate_specs[cand_idx], X_tr, y_tr)
                else:
                    from sklearn.linear_model import Ridge as _Ridge  # noqa: PLC0415
                    fitted = _Ridge(alpha=1.0, fit_intercept=True).fit(X_tr, y_tr)

                sq_errors = (y_va - fitted.predict(X_va)) ** 2
                fold_losses.append(float(np.mean(sq_errors)))

            avg_cv_loss = float(np.mean(fold_losses)) if fold_losses else float("inf")

        if avg_cv_loss < best_cv_loss:
            best_cv_loss = avg_cv_loss
            best_idx = cand_idx

    assert best_idx is not None, "No candidates provided."
    return best_idx
