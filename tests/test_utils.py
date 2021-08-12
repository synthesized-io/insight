import numpy as np
import pandas as pd

from src.synthesized_insight.metrics.utils import (bootstrap_statistic, bootstrap_binned_statistic,
                                                   permutation_test)


def test_bootstrap():
    def distance(h_x, h_y):
        return np.sum((h_x - h_y)**2)
    assert bootstrap_statistic((pd.Series([1]), pd.Series([0])), distance, n_samples=100).min() == 1
    assert bootstrap_statistic((pd.Series(range(2)), pd.Series(range(2, 4))), distance, 100).max() == 8


def test_bootstrap_binned():
    def distance(h_x, h_y):
        return np.linalg.norm(h_x - h_y, ord=1)

    assert bootstrap_binned_statistic((pd.Series([1, 3, 0]), pd.Series([1, 4, 3])), distance).max() == 12


def test_permutation():
    def distance(h_x, h_y):
        return abs(h_x.mean() - h_y.mean())
    assert permutation_test(pd.Series([1]), pd.Series([0]), distance).min() == 1

