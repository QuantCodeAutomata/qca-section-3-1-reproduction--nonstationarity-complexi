"""
download_factors.py
===================
Standalone script to download and cache all factor data needed for the
nonstationarity-complexity tradeoff experiments.

Run from the repository root::

    python data/download_factors.py

The script forces a full refresh of the Parquet cache located at
``data/cache/``.  Subsequent runs of any experiment script will load
from cache rather than re-downloading.
"""

import sys
import os

# Ensure the repository root is on sys.path so ``src`` is importable when
# the script is executed directly from any working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if __name__ == '__main__':
    from src.data_loader import load_full_panel

    X, y = load_full_panel(force_refresh=True)
    print(f'Predictor panel shape: {X.shape}')
    print(f'Target panel shape:    {y.shape}')
    print(f'Features: {list(X.columns[:5])} ...')
    print(f'Date range: {X.index[0]} to {X.index[-1]}')
