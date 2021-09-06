import numpy as np
import pandas as pd
import pytest

from src.synthesized_insight.check import Check, ColumnCheck
from src.synthesized_insight.metrics import (
    BinomialDistanceTest,
    EarthMoversDistance,
    HellingerDistance,
    KendallTauCorrelationTest,
    KolmogorovSmirnovDistanceTest,
    KruskalWallisTest,
    SpearmanRhoCorrelationTest,
)
from src.synthesized_insight.metrics.base import TwoColumnMetric
from src.synthesized_insight.metrics.statistical_tests import BootstrapTest, PermutationTest


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


class Distance(TwoColumnMetric):
    @classmethod
    def check_column_types(cls, sr_a: pd.Series, sr_b: pd.Series, check: Check = ColumnCheck()):
        return True

    def _compute_metric(self, sr_a, sr_b):
        metric_val = np.sum((sr_a - sr_b)**2)
        return metric_val


@pytest.fixture
def data1():
    return np.random.normal(0, 1, 1000)


@pytest.fixture
def data2():
    return np.random.normal(1, 1, 1000)


def test_distance(data1, data2):
    assert Distance()(pd.Series(data1), pd.Series(data2)) > 0


def test_pvalue(data1, data2):
    bootstrap_test = BootstrapTest(metric_cls_obj=Distance())
    _, p_value = bootstrap_test(pd.Series(data1), pd.Series(data2))
    assert p_value >= 0
    assert p_value <= 1


def test_binomial_distance_metric():
    metric = BinomialDistanceTest()
    data = (pd.Series([1, 0, 0]), pd.Series([1, 0, 1]))
    metric_value, p_value = metric(data[0], data[0])
    assert metric_value == 0
    assert p_value == 1

    metric_value, p_value = metric(data[0], data[1])
    assert np.abs(metric_value) > 0
    assert p_value <= 1
    assert p_value >= 0


def test_binomial_p_value():
    assert BinomialDistanceTest()(pd.Series([1, 1]), pd.Series([0, 0]))[1] == 0
    assert BinomialDistanceTest()(pd.Series([1, 0]), pd.Series([1, 0]))[1] == 1
    assert BinomialDistanceTest()(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 1, 0]))[1] == 0.625


def test_ks_distance():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series(['a', 'b', 'c', 'd'], name='b')

    ksd_test = KolmogorovSmirnovDistanceTest()
    assert ksd_test(sr_a, sr_a)[0] is not None
    assert ksd_test(sr_b, sr_b)[0] is None


def test_kt_correlation():
    np.random.seed(42)
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series(np.random.normal(0, 1, 5), name='b')
    sr_c = pd.Series(sr_b.values + np.random.normal(0, 0.8, 5), name='c')
    sr_d = pd.Series(['a', 'b', 'c', 'd'], name='d')

    kt_corr_test = KendallTauCorrelationTest(max_p_value=0.05)

    assert kt_corr_test(sr_a, sr_a)[0] is not None
    assert kt_corr_test(sr_b, sr_c)[0] is None
    assert kt_corr_test(sr_c, sr_d)[0] is None


def test_kt_correlation_string_numbers():
    sr_a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], name="a")
    sr_b = pd.Series(['1.4', '2.1', '', '4.1', '3.9', '4.4', '5.1', '6.0', '7.5', '9', '11.4', '12.1', '', '14.1', '13.9'], name="b")
    kendell_tau_correlation_test = KendallTauCorrelationTest(max_p_value=0.05)

    # df['TotalCharges'].dtype is Object in this case, eg. "103.4" instead of 103.4
    kt1 = kendell_tau_correlation_test(sr_a=sr_a, sr_b=sr_b)

    sr_b = pd.to_numeric(sr_b, errors='coerce')
    kt2 = kendell_tau_correlation_test(sr_a=sr_a, sr_b=sr_b)
    assert abs(kt1[0] - kt2[0]) < 0.01


def test_kruskal_wallis(group1):
    kruskal_wallis_test = KruskalWallisTest()
    assert kruskal_wallis_test(pd.Series(np.arange(10)), pd.Series(np.arange(10)))[0] == 0
    assert abs(kruskal_wallis_test(group1, group1)[0]) < 1e-6


def test_spearmans_rho():
    sr_a = pd.Series(np.random.normal(0, 1, 100), name='a')
    sr_b = pd.Series([1, 2, 3, 4, 5], name='b')
    sr_c = pd.Series([3, 5, 1, 3, 2], name='c')
    sr_d = pd.Series(['a', 'b', 'c', 'd'], name='d')

    spearman_rho_test = SpearmanRhoCorrelationTest(max_p_value=0.05)
    assert spearman_rho_test(sr_a, sr_a)[0] is not None
    assert spearman_rho_test(sr_b, sr_c)[0] is None
    assert spearman_rho_test(sr_c, sr_d)[0] is None


@pytest.mark.parametrize(
    'metric, data, alternative', [
        (EarthMoversDistance(), (pd.Series(['a', 'b', 'c']), pd.Series(['c', 'b', 'a'])), 'greater'),
        (HellingerDistance(), (pd.Series([1, 2, 3]), pd.Series([0, 0, 0])), 'less')
    ])
def test_bootstrap(metric, data, alternative):
    bootstrap = BootstrapTest(metric_cls_obj=metric)
    metric_val, p_value = bootstrap(data[0], data[0])
    assert metric_val == 0
    assert p_value == 1

    bootstrap = BootstrapTest(metric_cls_obj=metric, alternative=alternative)
    metric_val, p_value = bootstrap(data[1], data[1])
    assert metric_val == 0
    assert p_value == 1 if alternative != 'less' else p_value == 0

    _, p_value = bootstrap(data[0], data[1])
    assert p_value <= 1
    assert p_value >= 0


@pytest.mark.parametrize(
    'metric, data, alternative', [
        (EarthMoversDistance(), (pd.Series(['a', 'b', 'c']), pd.Series(['c', 'b', 'a'])), 'greater'),
        (HellingerDistance(), (pd.Series([1, 2, 3]), pd.Series([0, 0, 0])), 'less')
    ])
def test_permutation(metric, data, alternative):
    perm_test = PermutationTest(metric_cls_obj=metric)
    metric_val, p_value = perm_test(data[0], data[0])
    assert metric_val == 0
    assert p_value == 1

    perm_test = PermutationTest(metric_cls_obj=metric, alternative=alternative)
    metric_val, p_value = perm_test(data[1], data[1])
    assert metric_val == 0
    assert p_value == 1 if alternative != 'less' else p_value == 0

    _, p_value = perm_test(data[0], data[1])
    assert p_value <= 1
    assert p_value >= 0
