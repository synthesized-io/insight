from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from synthesized_insight.check import ColumnCheck
from synthesized_insight.metrics import (
    CramersV,
    DistanceCNCorrelation,
    DistanceNNCorrelation,
    EarthMoversDistance,
    EarthMoversDistanceBinned,
    HellingerDistance,
    JensenShannonDivergence,
    KendallTauCorrelationTest,
    KullbackLeiblerDivergence,
    Mean,
    Norm,
    R2Mcfadden,
    StandardDeviation,
)

mean = Mean()
std_dev = StandardDeviation()
cramers_v = CramersV()
emd = EarthMoversDistance()
kendell_tau_correlation_test = KendallTauCorrelationTest()
hellinger_distance = HellingerDistance()
distance_nn_corr = DistanceNNCorrelation()
distance_cn_corr = DistanceCNCorrelation()
kl_divergence = KullbackLeiblerDivergence()
js_divergence = JensenShannonDivergence()
r2_mcfadden = R2Mcfadden()
norm = Norm()
norm_ord1 = Norm(ord=1)


@pytest.fixture(scope='module')
def df():
    df = pd.read_csv("tests/datasets/mini_compas.csv")
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


@pytest.fixture
def data1():
    return np.random.normal(0, 1, 1000)


@pytest.fixture
def data2():
    return np.random.normal(1, 1, 1000)


def test_mean():
    sr_a = pd.Series(np.arange(100), name='a')
    val_a = mean(sr=sr_a)
    assert val_a == 49.5

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


def test_em_distance():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series(['a', 'b', 'c', 'd'], name='b')

    assert emd(sr_a, sr_a) is None
    assert emd(sr_b, sr_b) is not None


def test_cramers_v_basic():
    sr_a = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3] * 100, name='a')
    sr_b = pd.Series([1, 2, 3, 2, 3, 1, 3, 1, 2] * 100, name='b')

    assert cramers_v(sr_a, sr_a) > 0.99  # This metric -> cramers_v for large N (makes it more robust to outliers)
    assert cramers_v(sr_a, sr_b) == 0.

    sr_c = pd.Series(np.random.normal(0, 1, 1000), name='c')
    assert cramers_v(sr_c, sr_c) is None


def test_cramers_v_compas(df):
    check = ColumnCheck()
    continuous_cols, categorical_cols = [], []

    for col in df.columns:
        if check.continuous(df[col]):
            continuous_cols.append(col)
        elif check.categorical(df[col]):
            categorical_cols.append(col)

    for col_grp in combinations(categorical_cols, 2):
        assert(cramers_v(df[col_grp[0]], df[col_grp[1]]) is not None)


def test_repr():
    metric = EarthMoversDistance()
    assert repr(metric) == 'EarthMoversDistance()'


def test_str():
    metric = EarthMoversDistance()
    metric.name = 'earth_movers_distance'
    assert str(metric) == 'earth_movers_distance'


def test_basic_nn_distance_corr():
    sr_a = pd.Series(np.arange(10))
    sr_b = pd.Series(np.arange(10, 20))

    assert distance_nn_corr(sr_a, sr_b) == 1


def test_cn_basic_distance_corr():
    sr_a = pd.Series(np.random.choice(["A", "B"], size=10))
    sr_b = pd.Series(np.arange(10))
    assert distance_cn_corr(sr_a, sr_b) > 0


def test_nn_unequal_series_corr():
    sr_a = pd.Series(np.arange(20))
    sr_b = pd.Series(np.arange(10))

    assert distance_nn_corr(sr_a, sr_b) > 0


def test_cn_unequal_series_corr():
    sr_a = pd.Series(["A", "B", "A", "A", "B", "B", "C", "C", "C", "D", "D", "D", "E", "E", "F", "F", "F", "F"])
    sr_b = pd.Series([100, 200, 99, 101, 201, 199, 299, 300, 301, 500, 501, 505, 10, 12, 1001, 1050])
    assert distance_cn_corr(sr_a, sr_b) > 0.7


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

    assert norm(group1, group1) == 0
    assert norm(group1, group3) > norm(group1, group2)


def test_hellinger(group1, group2, group3):
    assert hellinger_distance(pd.Series([1, 0]), pd.Series([1, 0])) == 0
    assert hellinger_distance(pd.Series([1]), pd.Series([0])) == 1

    assert hellinger_distance(group1, group1) == 0
    assert hellinger_distance(group1, group3) > hellinger_distance(group1, group2)


def test_r2_mcfadden_correlation():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='x')
    sr_a_dates = pd.Series(pd.date_range('01/01/01', '01/12/01', name='x'))
    sr_b = pd.Series(np.random.choice([1, 0], 100), name='y')

    assert r2_mcfadden(sr_a, sr_a) is None  # continuous, continuous -> None

    assert r2_mcfadden(sr_b, sr_b) is None  # categorical, categorical -> None

    assert r2_mcfadden(sr_a_dates, sr_b) is None  # continuous date, categorical -> None

    assert r2_mcfadden(sr_b, sr_a) is not None  # categorical, continuous -> not None

    credit_df = pd.read_csv('tests/datasets/mini_credit.csv').sample(500)
    assert r2_mcfadden(sr_a=credit_df['SeriousDlqin2yrs'], sr_b=credit_df['age']) is not None
    assert r2_mcfadden(sr_a=credit_df['MonthlyIncome'], sr_b=credit_df['DebtRatio']) is None
    assert r2_mcfadden(sr_a=credit_df['SeriousDlqin2yrs'], sr_b=credit_df['MonthlyIncome']) is not None


def test_emd_distance_binned():

    def compare_and_log(x, y, bin_edges, val):
        emdb = EarthMoversDistanceBinned(bin_edges=bin_edges)
        metric_val = emdb(x, y)
        assert np.isclose(metric_val, val, rtol=0.1)

    compare_and_log(pd.Series([1, 2, 3]), pd.Series([1, 0, 3]), bin_edges=[0, 1, 2, 3], val=0.333)

    a = pd.Series(np.random.normal(loc=10, scale=1.0, size=10000))
    b = pd.Series(np.random.normal(loc=14, scale=1.0, size=10000))

    bin_edges = np.histogram_bin_edges(np.concatenate((a, b), axis=0), bins=100)
    x, _ = np.histogram(a, bins=bin_edges)
    y, _ = np.histogram(b, bins=bin_edges)
    compare_and_log(pd.Series(x), pd.Series(y), bin_edges, 4.0)
    compare_and_log(pd.Series([0, 3, 6, 14, 3]), pd.Series([1, 0, 8, 21, 1]), None, 0.20)
    compare_and_log(pd.Series([0, 3, 6, 14, 3]), pd.Series([0, 3, 6, 14, 3]), None, 0.0)
    compare_and_log(pd.Series([0, 0, 0, 0]), pd.Series([0, 3, 6, 14]), None, 1.0)
    compare_and_log(pd.Series([0, 0, 0, 0]), pd.Series([0, 0, 0, 0]), None, 0.0)
