import pytest

import numpy as np
import pandas as pd

from synthesized.insight.metrics.distance import DistanceMetric, EarthMoversDistance, HellingerDistance, KolmogorovSmirnovDistance, BinomialDistance, EarthMoversDistanceBinned, HellingerDistanceBinned


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


@pytest.mark.fast
def test_distance(data1, data2):
    assert Distance(data1, data2).distance > 0


@pytest.mark.fast
def test_pvalue(data1, data2):
    assert Distance(data1, data2).p_value >= 0
    assert Distance(data1, data2).p_value <= 1


@pytest.mark.fast
def test_interval(data1, data2):
    assert len(Distance(data1, data2).interval().value) == 2
    assert Distance(data1, data2).interval(0.95).level == 0.95


@pytest.mark.fast
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
