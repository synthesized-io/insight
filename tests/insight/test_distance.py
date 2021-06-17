import numpy as np
import pandas as pd
import pytest

from synthesized.insight.metrics.distance import (BinomialDistance, DistanceMetric, EarthMoversDistance,
                                                  EarthMoversDistanceBinned, HellingerDistance, HellingerDistanceBinned,
                                                  KolmogorovSmirnovDistance)


class Distance(DistanceMetric):
    @property
    def distance(self):
        return np.sum((self.x - self.y)**2)


@pytest.fixture
def data1():
    return np.random.normal(0, 1, 1000)


@pytest.fixture
def data2():
    return np.random.normal(1, 1, 1000)


def test_distance(data1, data2):
    assert Distance(data1, data2).distance > 0


def test_pvalue(data1, data2):
    assert Distance(data1, data2).p_value >= 0
    assert Distance(data1, data2).p_value <= 1


def test_interval(data1, data2):
    assert len(Distance(data1, data2).interval().value) == 2
    assert Distance(data1, data2).interval(0.95).level == 0.95


@pytest.mark.parametrize(
    'distance_metric, data, kwargs', [
        (KolmogorovSmirnovDistance, (pd.Series([1, 2, 3]), pd.Series([0, 0, 0])), {}),
        (EarthMoversDistance, (pd.Series([1, 2, 3]), pd.Series([0, 0, 0])), {}),
        (HellingerDistance, (pd.Series([1, 2, 3]), pd.Series([0, 0, 0])), {}),
        (BinomialDistance, (pd.Series([1, 0, 0]), pd.Series([1, 0, 1])), {}),
        (EarthMoversDistanceBinned, (pd.Series([1, 2, 3]), pd.Series([1, 0, 3])), {}),
        (HellingerDistanceBinned, (pd.Series([1, 2, 3]), pd.Series([1, 0, 3])), {'bins': [0, 1, 2, 3]})
    ])
def test_distance_metric(distance_metric, data, kwargs):

    metric = distance_metric(data[0], data[0], **kwargs)
    assert metric.distance == 0
    assert metric.p_value == 1

    metric = distance_metric(*data, **kwargs)
    assert np.abs(metric.distance) > 0
    assert metric.p_value <= 1
    assert metric.p_value >= 0


def test_emd_distance():

    def compare_and_log(x, y, bins, val):
        emd = EarthMoversDistanceBinned(x, y, bins)
        assert np.isclose(emd.distance, val, rtol=0.1)

    compare_and_log(pd.Series([1, 2, 3]), pd.Series([1, 0, 3]), bins=[0, 1, 2, 3], val=0.333)

    a = pd.Series(np.random.normal(loc=10, scale=1.0, size=10000))
    b = pd.Series(np.random.normal(loc=14, scale=1.0, size=10000))

    bins = np.histogram_bin_edges(np.concatenate((a, b), axis=0), bins=100)
    x, _ = np.histogram(a, bins=bins)
    y, _ = np.histogram(b, bins=bins)
    compare_and_log(x, y, bins, 4.0)
    compare_and_log(pd.Series([0, 3, 6, 14, 3]), pd.Series([1, 0, 8, 21, 1]), None, 0.20)
    compare_and_log(pd.Series([0, 3, 6, 14, 3]), pd.Series([0, 3, 6, 14, 3]), None, 0.0)
