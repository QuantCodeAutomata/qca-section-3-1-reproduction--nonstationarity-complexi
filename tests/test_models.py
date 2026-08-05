import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import CandidateModel, fit_candidate, compute_validation_errors


# ---------------------------------------------------------------------------
# CandidateModel.all_candidates()
# ---------------------------------------------------------------------------

def test_all_candidates_count():
    assert len(CandidateModel.all_candidates()) == 168


def test_all_candidates_have_unique_ids():
    candidates = CandidateModel.all_candidates()
    ids = [c.model_id for c in candidates]
    assert len(set(ids)) == len(ids)


def test_candidate_model_classes():
    classes = {c.model_class for c in CandidateModel.all_candidates()}
    assert classes == {"Ridge", "Lasso", "ElasticNet", "RF"}


# ---------------------------------------------------------------------------
# fit_candidate
# ---------------------------------------------------------------------------

def test_fit_candidate_ridge():
    rng = np.random.default_rng(10)
    X = rng.standard_normal((100, 5))
    true_coef = np.array([1.0, -2.0, 3.0, 0.0, 0.0])
    y = X @ true_coef + 0.1 * rng.standard_normal(100)

    ridge_candidates = [c for c in CandidateModel.all_candidates() if c.model_class == "Ridge"]
    model = fit_candidate(ridge_candidates[0], X[:80], y[:80])
    preds = model.predict(X[80:])

    corr = np.corrcoef(preds, y[80:])[0, 1]
    assert corr > 0.5


def test_fit_candidate_rf():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((150, 5))
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * rng.standard_normal(150)

    rf_candidates = [c for c in CandidateModel.all_candidates() if c.model_class == "RF"]
    model = fit_candidate(rf_candidates[0], X[:100], y[:100])
    preds = model.predict(X[100:])

    assert preds.shape == (50,)


# ---------------------------------------------------------------------------
# compute_validation_errors
# ---------------------------------------------------------------------------

def _fitted_model_and_val(rng_seed=99):
    rng = np.random.default_rng(rng_seed)
    X_train = rng.standard_normal((80, 5))
    y_train = rng.standard_normal(80)
    X_val = rng.standard_normal((20, 5))
    y_val = rng.standard_normal(20)
    candidate = CandidateModel.all_candidates()[0]
    model = fit_candidate(candidate, X_train, y_train)
    return model, X_val, y_val


def test_validation_errors_nonnegative():
    model, X_val, y_val = _fitted_model_and_val()
    errors = compute_validation_errors(model, X_val, y_val)
    assert (errors >= 0).all()


def test_validation_errors_shape():
    model, X_val, y_val = _fitted_model_and_val()
    errors = compute_validation_errors(model, X_val, y_val)
    assert len(errors) == len(y_val)
