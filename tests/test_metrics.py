from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from synthesized_insight.check import ColumnCheck
from synthesized_insight.metrics import (
    BhattacharyyaCoefficient,
    CramersV,
    EarthMoversDistance,
    EarthMoversDistanceBinned,
    HellingerDistance,
    JensenShannonDivergence,
    KullbackLeiblerDivergence,
    Mean,
    Norm,
    StandardDeviation,
    TotalVariationDistance,
)

mean = Mean()
std_dev = StandardDeviation()
cramers_v = CramersV()
emd = EarthMoversDistance()
hellinger_distance = HellingerDistance()
kl_divergence = KullbackLeiblerDivergence()
js_divergence = JensenShannonDivergence()
norm = Norm()
norm_ord1 = Norm(ord=1)
bhattacharyya_coefficient = BhattacharyyaCoefficient()
total_variation_distance = TotalVariationDistance()


@pytest.fixture(scope='module')
def df():
    df = pd.read_csv("https://raw.githubusercontent.com/synthesized-io/datasets/master/tabular/biased/compas.csv")
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


def test_bhattacharyya_coefficient_complete_overlap(group1):
    """
    Tests that the BhattacharyyaCoefficient metric yields the correct result with completely overlapping distributions.
    """
    assert np.isclose(bhattacharyya_coefficient(pd.Series([1, 0]), pd.Series([1, 0])), 1)
    assert np.isclose(bhattacharyya_coefficient(group1, group1), 1)


def test_bhattacharyya_coefficient_no_overlap():
    """
    Tests that the BhattacharyyaCoefficient metric yields the correct result when the two distributions do not overlap.
    """
    assert np.isclose(bhattacharyya_coefficient(pd.Series([1]), pd.Series([0])), 0)


def test_bhattacharyya_coefficient_inequality_preserved(group1, group2, group3):
    """
    Tests that BhattacharyyaCoefficient metric preserves inequality when two groups of distributions have differing
    overlap.
    """
    assert bhattacharyya_coefficient(group1, group3) < bhattacharyya_coefficient(group1, group2)


def test_bhattacharyya_coefficient_hellinger_distance_relation(group1, group2, group3):
    """
    Tests that the BhattacharyyaCoefficient conforms to its relationship with hellinger_distance.
    """
    assert np.isclose(bhattacharyya_coefficient(group1, group3), 1 - hellinger_distance(group1, group3)**2)


def test_total_variation_distance_complete_overlap(group1, group2, group3):
    """
    Tests that the TotalVariation distance yields the correct result when the distributions completely overlap.
    """
    assert np.isclose(total_variation_distance(pd.Series([1, 0]), pd.Series([1, 0])), 0)
    assert np.isclose(total_variation_distance(group1, group1), 0)


def test_total_variation_distance_no_overlap():
    """
    Tests that the TotalVariation distance yields the correct result when the distributions do not overlap.
    """
    assert np.isclose(total_variation_distance(pd.Series([1]), pd.Series([0])), 1)


def test_total_variation_distance_inequality_preserved(group1, group2, group3):
    """
    Tests that the TotalVariation distance preserves inequality when two groups of distributions have differing overlap.
    """
    assert total_variation_distance(group1, group3) > total_variation_distance(group1, group2)


def test_total_variation_distance_hellinger_inequality_preserved(group1, group2, group3):
    """
    Tests that the TotalVariation distance preserves its inequality relationship with hellinger distance.
    """
    assert total_variation_distance(group1, group3) > hellinger_distance(group1, group3) ** 2
    assert total_variation_distance(group1, group3) < hellinger_distance(group1, group3) * np.sqrt(2)
