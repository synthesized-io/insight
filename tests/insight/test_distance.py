import pytest

import numpy as np

from synthesized.insight.metrics.distance import DistanceMetric


class Distance(DistanceMetric):
    @property
    def distance(self):
        return np.sum((self.x - self.y)**2)


@pytest.fixture
def data1():
    return np.random.normal(0, 1, 1000)


@pytest.fixture
def data2():
    return np.random.normal(0, 1, 1000)


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
