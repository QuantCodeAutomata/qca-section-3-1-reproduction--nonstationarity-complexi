import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.trading import compute_trading_signals, compute_wealth_path, compute_terminal_wealth_excess


# ---------------------------------------------------------------------------
# compute_trading_signals
# ---------------------------------------------------------------------------

def test_trading_signals_positive():
    assert compute_trading_signals(pd.Series([0.5, 1.0])).tolist() == [1, 1]


def test_trading_signals_negative():
    assert compute_trading_signals(pd.Series([-0.5, -1.0])).tolist() == [-1, -1]


def test_trading_signals_zero():
    assert compute_trading_signals(pd.Series([0.0])).tolist() == [0]


# ---------------------------------------------------------------------------
# compute_wealth_path
# ---------------------------------------------------------------------------

def test_wealth_path_constant_positive_returns():
    # wealth[t] = Π_{s=0}^{t} (1 + signal[s] * return[s])
    signals = pd.Series([1, 1, 1], index=range(3))
    returns = pd.Series([0.01, 0.01, 0.01], index=range(3))
    wealth = compute_wealth_path(signals, returns)
    assert wealth.iloc[-1] == pytest.approx(1.01 ** 3)


def test_wealth_path_initial_value():
    signals = pd.Series([1], index=[0])
    returns = pd.Series([0.05], index=[0])
    wealth = compute_wealth_path(signals, returns)
    assert wealth.iloc[0] == pytest.approx(1.0 * (1 + 0.05 * 1))


def test_wealth_path_nonnegative():
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.uniform(-0.5, 0.5, 20), index=range(20))
    signals = pd.Series(np.ones(20, dtype=int), index=range(20))
    wealth = compute_wealth_path(signals, returns)
    assert (wealth >= 0).all()


# ---------------------------------------------------------------------------
# compute_terminal_wealth_excess
# ---------------------------------------------------------------------------

def test_terminal_wealth_excess_positive():
    dates = pd.date_range("2000-01", periods=3, freq="ME")
    wealth_atoms = pd.Series([1.0, 1.5, 2.0], index=dates)
    wealth_baseline = pd.Series([1.0, 1.2, 1.5], index=dates)
    assert compute_terminal_wealth_excess(wealth_atoms, wealth_baseline) > 0


def test_terminal_wealth_excess_negative():
    dates = pd.date_range("2000-01", periods=3, freq="ME")
    wealth_atoms = pd.Series([1.0, 1.0, 0.8], index=dates)
    wealth_baseline = pd.Series([1.0, 1.2, 1.5], index=dates)
    assert compute_terminal_wealth_excess(wealth_atoms, wealth_baseline) < 0


def test_terminal_wealth_excess_formula():
    dates = pd.date_range("2000-01", periods=3, freq="ME")
    wealth_atoms = pd.Series([1.0, 1.5, 2.0], index=dates)
    wealth_baseline = pd.Series([1.0, 1.2, 1.5], index=dates)
    excess = compute_terminal_wealth_excess(wealth_atoms, wealth_baseline)
    assert excess == pytest.approx(2.0 / 1.5 - 1)
