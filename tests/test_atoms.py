import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.atoms import (
    compute_delta_hat,
    compute_psi_hat,
    compute_phi_hat,
    select_validation_horizon,
    atoms_pairwise_compare,
    run_atoms_tournament,
    select_fixed_val,
    DELTA_PRIME,
    M_SQUARED,
)


# ---------------------------------------------------------------------------
# compute_delta_hat
# ---------------------------------------------------------------------------

def test_delta_hat_mean():
    u_values = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_delta_hat(u_values) == pytest.approx(2.5)


def test_delta_hat_single_value():
    u_values = np.array([5.0])
    assert compute_delta_hat(u_values) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# compute_psi_hat
# ---------------------------------------------------------------------------

def test_psi_hat_single_observation():
    # n=1, zero deviation from mean ⟹ ψ̂ = 8·M²
    assert compute_psi_hat(np.array([1.0])) == pytest.approx(8 * M_SQUARED)


def test_psi_hat_positive():
    rng = np.random.default_rng(7)
    assert compute_psi_hat(rng.standard_normal(50)) > 0


def test_psi_hat_decreases_with_more_data():
    # Same constant signal — only the 1/n factor changes
    psi10 = compute_psi_hat(np.zeros(10) + 0.1)
    psi100 = compute_psi_hat(np.zeros(100) + 0.1)
    assert psi10 > psi100


# ---------------------------------------------------------------------------
# compute_phi_hat
# ---------------------------------------------------------------------------

def test_phi_hat_nonnegative():
    delta_hats = [0.1, 0.2, 0.15]
    psi_hats = [0.05, 0.05, 0.05]
    phi = compute_phi_hat(delta_hats, psi_hats)
    assert all(p >= 0 for p in phi)


def test_phi_hat_length_matches_input():
    delta_hats = [0.1, 0.2, 0.15]
    psi_hats = [0.05, 0.05, 0.05]
    phi = compute_phi_hat(delta_hats, psi_hats)
    assert len(phi) == len(delta_hats)


# ---------------------------------------------------------------------------
# atoms_pairwise_compare
# ---------------------------------------------------------------------------

def test_pairwise_compare_f1_wins_when_better():
    # f1 has much smaller errors → f1 should win
    val_f1 = {0: np.array([0.01, 0.01]), 1: np.array([0.01])}
    val_f2 = {0: np.array([0.10, 0.10]), 1: np.array([0.10])}
    winner, _ = atoms_pairwise_compare(val_f1, val_f2, available_horizons=2)
    assert winner == 1


def test_pairwise_compare_f2_wins_when_better():
    # f2 has much smaller errors → f2 should win
    val_f1 = {0: np.array([0.10, 0.10]), 1: np.array([0.10])}
    val_f2 = {0: np.array([0.01, 0.01]), 1: np.array([0.01])}
    winner, _ = atoms_pairwise_compare(val_f1, val_f2, available_horizons=2)
    assert winner == 2


# ---------------------------------------------------------------------------
# run_atoms_tournament
# ---------------------------------------------------------------------------

def test_tournament_returns_single_winner():
    n_candidates = 5
    available_horizons = 2
    rng = np.random.default_rng(42)
    val_errors_by_candidate_month = {
        i: {h: rng.random(10) for h in range(available_horizons)}
        for i in range(n_candidates)
    }
    candidates = list(range(n_candidates))
    winner, _ = run_atoms_tournament(
        candidates, val_errors_by_candidate_month, available_horizons
    )
    assert isinstance(winner, (int, np.integer))
    assert winner in range(n_candidates)


def test_tournament_best_wins_dominated_case():
    # Candidate 0 has near-zero errors; all others have near-1.0 errors
    n_candidates = 5
    available_horizons = 2
    val_errors_by_candidate_month = {}
    for i in range(n_candidates):
        err = 0.001 if i == 0 else 1.0
        val_errors_by_candidate_month[i] = {
            h: np.full(12, err) for h in range(available_horizons)
        }
    candidates = list(range(n_candidates))
    winner, _ = run_atoms_tournament(
        candidates, val_errors_by_candidate_month, available_horizons
    )
    assert winner == 0


# ---------------------------------------------------------------------------
# select_fixed_val
# ---------------------------------------------------------------------------

def test_select_fixed_val_picks_minimum_sse():
    # Candidate 1 always has the smallest errors across all horizons
    n_candidates = 3
    available_horizons = 5
    window_months = 12
    val_errors_by_candidate_month = {}
    for i in range(n_candidates):
        err = 0.001 if i == 1 else 1.0
        val_errors_by_candidate_month[i] = {
            h: np.full(10, err) for h in range(available_horizons)
        }
    winner = select_fixed_val(
        val_errors_by_candidate_month, available_horizons, window_months
    )
    assert winner == 1
