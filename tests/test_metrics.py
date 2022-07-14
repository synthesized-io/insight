from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import yaml

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


def test_metrics_to_yaml_dump(capsys):
    """
    Tests that the metrics are dumped into the standard output properly.
    """
    metrics = [mean, std_dev, cramers_v, norm_ord1]
    Mean.metrics_to_yaml_dump(metrics)
    out = capsys.readouterr().out
    expected = {'metrics': [
                    {'name': 'mean'},
                    {'name': 'standard_deviation', 'remove_outliers': 0.0},
                    {'name': 'cramers_v'},
                    {'name':'norm', 'ord': 1}]}

    assert out.strip() == yaml.dump(data=expected).strip()

def test_metrics_from_yaml(capsys):
    """
    Tests that the metrics are restored form a yaml properly.
    """
    metrics = [mean, std_dev, cramers_v, norm_ord1]
    Mean.metrics_to_yaml_dump(metrics)
    out = capsys.readouterr().out.strip()
    new_metrics = Mean.metrics_from_yaml(out)

    assert len(new_metrics) == 4

    for old, new in zip(metrics, new_metrics):
        assert isinstance(new, type(old))

    assert np.isclose(new_metrics[1].remove_outliers, 0)
    assert new_metrics[3].ord == 1


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


def test_base_to_dict():
    """
    Tests the basic variation of _Metric.to_dict().
    """
    dict_mean = mean.to_dict()
    assert dict_mean['name'] == 'mean'

    dict_cramers_v = cramers_v.to_dict()
    assert dict_cramers_v['name'] == 'cramers_v'

    dict_emd = emd.to_dict()
    assert dict_emd['name'] == 'earth_movers_distance'

    dict_kl_divergence = kl_divergence.to_dict()
    assert dict_kl_divergence['name'] == 'kullback_leibler_divergence'

    dict_js_divergence = js_divergence.to_dict()
    assert dict_js_divergence['name'] == 'jensen_shannon_divergence'

    dict_hellinger_distance = hellinger_distance.to_dict()
    assert dict_hellinger_distance['name'] == 'hellinger_distance'

    dict_bc_coef = bhattacharyya_coefficient.to_dict()
    assert dict_bc_coef['name'] == 'bhattacharyya_coefficient'


def test_base_from_dict():
    """
    Tests the basic variation of _Metric.from_dict.
    """
    dict_mean = {'name': 'mean'}
    new_mean = Mean.from_dict(dict_mean)
    assert isinstance(new_mean, Mean)

    dict_cramers_v = {'name': 'cramers_v'}
    new_cramers_v = CramersV.from_dict(dict_cramers_v)
    assert isinstance(new_cramers_v, CramersV)

    dict_emd = {'name': 'earth_movers_distance'}
    new_emd = EarthMoversDistance.from_dict(dict_emd)
    assert isinstance(new_emd, EarthMoversDistance)

    dict_kl_divergence = {'name': 'kullback_leibler_divergence'}
    new_kl_divergence = KullbackLeiblerDivergence.from_dict(dict_kl_divergence)
    assert isinstance(new_kl_divergence, KullbackLeiblerDivergence)

    dict_js_divergence = {'name': 'jensen_shannon_divergence'}
    new_js_divergence = JensenShannonDivergence.from_dict(dict_js_divergence)
    assert isinstance(new_js_divergence, JensenShannonDivergence)

    dict_hellinger_distance = {'name': 'hellinger_distance'}
    new_hellinger_distance = HellingerDistance.from_dict(dict_hellinger_distance)
    assert isinstance(new_hellinger_distance, HellingerDistance)

    dict_bc_coef = {'name': 'bhattacharyya_coefficient'}
    new_bc_coef = BhattacharyyaCoefficient.from_dict(dict_bc_coef)
    assert isinstance(new_bc_coef, BhattacharyyaCoefficient)


def test_from_dict_different_class():
    dict_mean = {'name': 'mean'}
    new_mean = BhattacharyyaCoefficient.from_dict(dict_mean)
    assert isinstance(new_mean, Mean)

    dict_norm = {'name': 'norm', 'ord': 1}
    new_norm = HellingerDistance.from_dict(dict_norm)

    assert isinstance(new_norm, Norm)
    assert new_norm.ord == 1


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


def test_standard_deviation_to_dict():
    """
    Tests the to_dict method that is specific to StandardDeviation metric.
    """
    dict_std_dev = std_dev.to_dict()
    assert dict_std_dev['name'] == 'standard_deviation'
    assert np.isclose(dict_std_dev['remove_outliers'], 0)


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
        assert (cramers_v(df[col_grp[0]], df[col_grp[1]]) is not None)


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


def test_norm_to_dict_ord_one():
    """
    Tests the to_dict method that is specific to the Norm metric. Tests that to_dict takes account of the changed ord
    parameter.
    """
    dict_norm_one = norm_ord1.to_dict()
    assert dict_norm_one['name'] == 'norm'
    assert np.isclose(dict_norm_one['ord'], 1)


def test_norm_to_dict_ord_default():
    """
    Tests the to_dict method that is specific to the Norm metric. Tests that to_dict takes account of the default ord
    parameter.
    """
    dict_norm_two = norm.to_dict()
    assert dict_norm_two['name'] == 'norm'
    assert np.isclose(dict_norm_two['ord'], 2)


def test_norm_from_dict_ord_one():
    dict_norm_one = {'name': 'norm', 'ord': 1}
    new_norm_one = Norm.from_dict(dict_norm_one)

    assert isinstance(new_norm_one, Norm)
    assert new_norm_one.ord == 1


def test_norm_from_dict_ord_default():
    dict_norm = {'name': 'norm'}
    new_norm = Norm.from_dict(dict_norm)

    assert isinstance(new_norm, Norm)
    assert new_norm.ord == 2


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


def test_emd_distance_binned_no_bins_to_dict():
    """
    Tests that the EarthMoversDistanceBinned metric has a properly functioning to_dict method when the default
    parameters are untouched.
    """
    emdb = EarthMoversDistanceBinned()
    dict_emdb = emdb.to_dict()
    assert dict_emdb['name'] == 'earth_movers_distance_binned'
    assert dict_emdb['bin_edges'] is None


def test_emd_distance_binned_no_bins_from_dict():
    """
    Tests that the binned earth mover's distance can be successfully be built from a dictionary when no bin edges are
    specified.
    """
    dict_emdb = {'name': 'earth_movers_distance_binned'}

    new_emdb = EarthMoversDistanceBinned.from_dict(dict_emdb)
    assert isinstance(new_emdb, EarthMoversDistanceBinned)


def test_emd_distance_binned_to_dict():
    """
    Tests that the EarthMoversDistanceBinned metric has a properly functioning to_dict method when the default parameter
    is specified.
    """
    a = pd.Series(np.random.normal(loc=10, scale=1.0, size=10000))
    bin_edges = np.histogram_bin_edges(a, bins=100)

    emdb = EarthMoversDistanceBinned(bin_edges=bin_edges)
    dict_emdb = emdb.to_dict()
    assert dict_emdb['name'] == 'earth_movers_distance_binned'
    assert np.allclose(dict_emdb['bin_edges'], bin_edges)


def test_emd_distance_binned_from_dict():
    """
    Tests that the binned earth mover's distance can be successfully be built from a dictionary with specified
    bin edges.
    """
    a = pd.Series(np.random.normal(loc=10, scale=1.0, size=10000))
    bin_edges = np.histogram_bin_edges(a, bins=100)
    dict_emdb = {'name': 'earth_movers_distance_binned', 'bin_edges': bin_edges}
    new_emdb = EarthMoversDistanceBinned.from_dict(dict_emdb)

    assert isinstance(new_emdb, EarthMoversDistanceBinned)
    assert np.allclose(new_emdb.bin_edges, bin_edges)


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
    assert np.isclose(bhattacharyya_coefficient(group1, group3), 1 - hellinger_distance(group1, group3) ** 2)


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


def test_total_variation_distance_hellinger_inequality_preserved(group1, group3):
    """
    Tests that the TotalVariation distance preserves its inequality relationship with hellinger distance.
    """
    assert total_variation_distance(group1, group3) > hellinger_distance(group1, group3) ** 2
    assert total_variation_distance(group1, group3) < hellinger_distance(group1, group3) * np.sqrt(2)
