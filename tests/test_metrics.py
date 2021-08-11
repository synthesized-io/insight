import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

import pytest

from src.synthesized_insight.metrics import (CramersV, EarthMoversDistance, HellingerDistance, JensenShannonDivergence,
                                             KendallTauCorrelation, KolmogorovSmirnovDistance, Mean, Norm,
                                             SpearmanRhoCorrelation, StandardDeviation, R2Mcfadden, KruskalWallis,
                                             DistanceNNCorrelation, DistanceCNCorrelation, KullbackLeiblerDivergence,
                                             BinomialDistance)

mean = Mean()
std_dev = StandardDeviation()
cramers_v = CramersV()
emd = EarthMoversDistance()
ksd = KolmogorovSmirnovDistance()
kendell_tau_correlation = KendallTauCorrelation()
hellinger_distance = HellingerDistance()
distance_nn_corr = DistanceNNCorrelation()
distance_cn_corr = DistanceCNCorrelation()
kl_divergence = KullbackLeiblerDivergence()
js_divergence = JensenShannonDivergence()
r2_mcfadden = R2Mcfadden()
kruskal_wallis = KruskalWallis()
norm = Norm()
norm_ord1 = Norm(ord=1)
binomial_distance = BinomialDistance()


@pytest.fixture(scope='module')
def df():
    df = pd.read_csv("datasets/compas.csv")
    return df


@pytest.fixture(scope='module')
def group1(df):
    pred1 = df["Ethnicity"] == "Caucasian"
    target_attr = "RawScore"
    group1 = df[pred1][target_attr]
    return group1


@pytest.fixture(scope='module')
def group2(df):
    pred2 = df["Ethnicity"] == "African-American"
    target_attr = "RawScore"
    group2 = df[pred2][target_attr]
    return group2


@pytest.fixture(scope='module')
def group3(group2):
    group3 = group2.sort_values()[len(group2) // 2:]
    return group3


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


def test_binomial_distance():
    assert binomial_distance(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert binomial_distance(pd.Series([1, 1]), pd.Series([0, 0])) == 1
    assert binomial_distance(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 0, 0])) == 0.5

    assert binomial_distance(pd.Series([True, False]), pd.Series([True, False])) == 0
    assert binomial_distance(pd.Series([False, False]), pd.Series([True, True])) == -1
    assert binomial_distance(pd.Series([True, False, True, True]), pd.Series([True, False, False, False])) == 0.5


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


def test_basic_nn_distance_corr():
    sr_a = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    sr_b = pd.Series([30.0, 10.0, 20.0, 60.0, 50.0, 40.0])

    assert distance_nn_corr(sr_a, sr_b) > 0.75


def test_cn_basic_distance_corr():
    sr_a = pd.Series(["A", "B", "A", "A", "B", "B"])
    sr_b = pd.Series([15, 45, 14, 16, 44, 46])

    assert distance_cn_corr(sr_a, sr_b) > 0.8


def test_nn_unequal_series_corr():
    sr_a = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    sr_b = pd.Series([10.0, 20.0, 60.0])

    assert distance_nn_corr(sr_a, sr_b) > 0.7


def test_cn_unequal_series_corr():
    sr_a = pd.Series(["A", "B", "A", "A", "B", "B", "C", "C", "C", "D", "D", "D", "E", "E", "F", "F", "F", "F"])
    sr_b = pd.Series([100, 200, 99, 101, 201, 199, 299, 300, 301, 500, 501, 505, 10, 12, 1001, 1050])

    assert distance_cn_corr(sr_a, sr_b) > 0.7


def test_kruskal_wallis(group1):
    assert kruskal_wallis(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert abs(kruskal_wallis(group1, group1)) < 1e-6


def test_kl_divergence(group1):
    assert kl_divergence(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert kl_divergence(pd.Series([1, 1]), pd.Series([0, 0])) == float("inf")

    assert kl_divergence(group1, group1) == 0


def test_js_divergence(group1, group2, group3):
    assert js_divergence(pd.Series([1, 0]), pd.Series([1, 0])) == 0

    assert js_divergence(group1, group1) == 0
    assert js_divergence(group1, group3) > js_divergence(group1, group2)


def test_norm(group1, group2, group3):
    assert norm(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert norm_ord1(pd.Series([1]), pd.Series([0])) == 2
    assert norm_ord1(pd.Series(np.arange(5)), pd.Series(np.arange(5, 10))) == 2

    assert norm(group1, group1) == 0
    assert norm(group1, group3) > Norm()(group1, group2)


# def test_hellinger(group1, group2, group3):
#     assert hellinger_distance(pd.Series([1, 0]), pd.Series([1, 0])) == 0
#     assert hellinger_distance(pd.Series([1]), pd.Series([0])) == 1

#     assert hellinger_distance(group1, group1) == 0
#     assert hellinger_distance(group1, group3) > hellinger_distance(group1, group2)

