"""Master script: runs all four experiments in sequence."""

import sys
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

from src.data_loader import load_full_panel
from exp.exp1_complexity_tradeoff import run_exp1, analyze_exp1
from exp.exp2_atoms_vs_baselines import run_exp2, analyze_exp2
from exp.exp3_recession_analysis import run_exp3
from exp.exp4_trading_strategy import run_exp4

if __name__ == '__main__':
    print('Loading data...')
    X, y = load_full_panel()

    print('\n=== Experiment 1: Complexity-Nonstationarity Tradeoff (fast mode) ===')
    results1 = run_exp1(X, y, fast_mode=True)
    analyze_exp1(results1)

    print('\n=== Experiment 2: ATOMS vs Baselines (fast mode) ===')
    results2 = run_exp2(X, y, fast_mode=True)
    analyze_exp2(results2)

    print('\n=== Experiment 3: Recession Analysis ===')
    results3 = run_exp3(X, y, fast_mode=True)
    print(results3)

    print('\n=== Experiment 4: Trading Strategy ===')
    results4 = run_exp4(X, y, fast_mode=True)
    print(results4)

    print('\nAll experiments complete. Results saved to results/')
