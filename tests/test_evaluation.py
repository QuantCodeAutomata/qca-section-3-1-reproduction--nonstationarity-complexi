import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import compute_oos_r2, compute_annual_r2, compute_recession_r2


# ---------------------------------------------------------------------------
# compute_oos_r2
# ---------------------------------------------------------------------------

def test_oos_r2_perfect_prediction():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 4, 5]
    assert compute_oos_r2(y_true, y_pred) == 1.0


def test_oos_r2_zero_prediction():
    # y_pred = 0 ⟹ SSE = Σ y_true² = denom ⟹ R² = 0
    y_true = [1, 2, 3]
    y_pred = [0, 0, 0]
    assert compute_oos_r2(y_true, y_pred) == pytest.approx(0.0)


def test_oos_r2_can_be_negative():
    # Very bad predictor: SSE ≫ denom → R² ≪ 0
    assert compute_oos_r2(np.array([1, 1, 1]), np.array([10, 10, 10])) < 0


def test_oos_r2_zero_benchmark():
    # Zero-benchmark denominator is Σ y², NOT variance.
    # y_pred = 0  ⟹  R² = 1 - Σy²/Σy² = 0
    y_true = np.array([1, -1, 1, -1])
    y_pred = np.zeros(4)
    assert abs(compute_oos_r2(y_true, y_pred)) < 1e-10


# ---------------------------------------------------------------------------
# compute_annual_r2
# ---------------------------------------------------------------------------

def test_annual_r2_returns_series():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-31", "2002-12-31", freq="ME")
    y_true = pd.Series(rng.standard_normal(len(dates)), index=dates)
    y_pred = pd.Series(rng.standard_normal(len(dates)), index=dates)

    result = compute_annual_r2(y_true, y_pred)

    assert isinstance(result, pd.Series)
    assert all(isinstance(idx, (int, np.integer)) for idx in result.index)
    assert all(isinstance(float(v), float) for v in result.values)


def test_annual_r2_single_year():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2000-01-31", "2000-12-31", freq="ME")
    y_true = pd.Series(rng.standard_normal(len(dates)), index=dates)
    y_pred = pd.Series(rng.standard_normal(len(dates)), index=dates)

    result = compute_annual_r2(y_true, y_pred)

    assert len(result) == 1
    assert 2000 in result.index


# ---------------------------------------------------------------------------
# compute_recession_r2
# ---------------------------------------------------------------------------

def test_recession_r2_returns_dict_with_expected_keys():
    rng = np.random.default_rng(2)
    n = 100
    dates = pd.date_range("1990-01-31", periods=n, freq="ME")
    y_true = pd.Series(rng.standard_normal(n), index=dates)
    y_pred = pd.Series(rng.standard_normal(n), index=dates)

    gulf_war = pd.Series(np.zeros(n, dtype=bool), index=dates)
    dot_com  = pd.Series(np.zeros(n, dtype=bool), index=dates)
    gfc      = pd.Series(np.zeros(n, dtype=bool), index=dates)
    full_oos = pd.Series(np.ones(n,  dtype=bool), index=dates)
    gulf_war.iloc[:10]  = True
    dot_com.iloc[20:30] = True
    gfc.iloc[50:70]     = True

    masks = {"gulf_war": gulf_war, "dot_com": dot_com, "gfc": gfc, "full_oos": full_oos}
    result = compute_recession_r2(y_true, y_pred, masks)

    assert set(result.keys()) == {"gulf_war", "dot_com", "gfc", "full_oos"}
