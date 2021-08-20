import numpy as np
import pandas as pd

from src.synthesized_insight.metrics import (
    CramersV,
    EarthMoversDistance,
    KendallTauCorrelation,
    KolmogorovSmirnovDistance,
    Mean,
    SpearmanRhoCorrelation,
    StandardDeviation,
)

mean = Mean()
std_dev = StandardDeviation()
cramers_v = CramersV()
emd = EarthMoversDistance()
ksd = KolmogorovSmirnovDistance()
kendell_tau_correlation = KendallTauCorrelation()


def test_mean():
    sr_a = pd.Series([1, 2, 3, 4, 5], name='a')
    val_a = mean(sr=sr_a)
    assert val_a == 3

    sr_b = pd.Series(np.datetime64('2020-01-01') + np.arange(0, 3, step=1).astype('m8[D]'), name='b')
    val_b = mean(sr=sr_b)
    assert val_b == np.datetime64('2020-01-02')

    sr_c = pd.Series(['a', 'b', 'c', 'd'], name='c')
    val_c = mean(sr=sr_c)
    assert val_c is None


def test_standard_deviation():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    val_a = std_dev(sr=sr_a)
    assert val_a is not None

    sr_b = pd.Series(np.datetime64('2020-01-01') + np.arange(0, 20, step=1).astype('m8[D]'), name='b')
    val_b = std_dev(sr=sr_b)
    assert val_b is not None

    sr_c = pd.Series(['a', 'b', 'c', 'd'], name='c')
    val_c = std_dev(sr=sr_c)
    assert val_c is None


def test_spearmans_rho():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series([1, 2, 3, 4, 5], name='b')
    sr_c = pd.Series([3, 5, 1, 3, 2], name='c')
    sr_d = pd.Series(['a', 'b', 'c', 'd'], name='d')

    spearman_rho = SpearmanRhoCorrelation(max_p_value=0.05)

    assert spearman_rho(sr_a, sr_a) is not None
    assert spearman_rho(sr_b, sr_c) is None
    assert spearman_rho(sr_c, sr_d) is None


def test_em_distance():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series(['a', 'b', 'c', 'd'], name='b')

    assert emd(sr_a, sr_a) is None
    assert emd(sr_b, sr_b) is not None


def test_ks_distance():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series(['a', 'b', 'c', 'd'], name='b')

    assert ksd(sr_a, sr_a) is not None
    assert ksd(sr_b, sr_b) is None


def test_kt_correlation():
    np.random.seed(42)
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series(np.random.normal(0, 1, 5), name='b')
    sr_c = pd.Series(sr_b.values + np.random.normal(0, 0.8, 5), name='c')
    sr_d = pd.Series(['a', 'b', 'c', 'd'], name='d')

    kt_corr = KendallTauCorrelation(max_p_value=0.05)

    assert kt_corr(sr_a, sr_a) is not None
    assert kt_corr(sr_b, sr_c) is None
    assert kt_corr(sr_c, sr_d) is None


def test_cramers_v():
    sr_a = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3] * 100, name='a')
    sr_b = pd.Series([1, 2, 3, 2, 3, 1, 3, 1, 2] * 100, name='b')

    assert cramers_v(sr_a, sr_a) > 0.99  # This metric -> cramers_v for large N (makes it more robust to outliers)
    assert cramers_v(sr_a, sr_b) == 0.

    sr_c = pd.Series(np.random.normal(0, 1, 1000), name='c')
    assert cramers_v(sr_c, sr_c) is None


def test_kt_correlation_string_numbers():
    sr_a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], name="a")
    sr_b = pd.Series(['1.4', '2.1', '', '4.1', '3.9', '4.4', '5.1', '6.0', '7.5', '9', '11.4', '12.1', '', '14.1', '13.9'], name="b")

    # df['TotalCharges'].dtype is Object in this case, eg. "103.4" instead of 103.4
    kt1 = kendell_tau_correlation(sr_a=sr_a, sr_b=sr_b)

    sr_b = pd.to_numeric(sr_b, errors='coerce')
    kt2 = kendell_tau_correlation(sr_a=sr_a, sr_b=sr_b)
    assert abs(kt1 - kt2) < 0.01


def test_repr():
    metric = KolmogorovSmirnovDistance()
    assert repr(metric) == 'KolmogorovSmirnovDistance()'


def test_str():
    metric = KolmogorovSmirnovDistance()
    metric.name = 'kolmogorov_smirnov_distance'
    assert str(metric) == 'kolmogorov_smirnov_distance'
