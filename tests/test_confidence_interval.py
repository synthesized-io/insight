import pandas as pd
import pytest

from synthesized_insight.metrics import EarthMoversDistance, HellingerDistance
from synthesized_insight.metrics.confidence_interval import compute_binomial_interval, compute_bootstrap_interval


@pytest.mark.parametrize(
    'metric, data', [
        (EarthMoversDistance(), (pd.Series(['a', 'b', 'c']), pd.Series(['c', 'b', 'a']))),
        (HellingerDistance(), (pd.Series([1, 2, 3]), pd.Series([0, 0, 0])))
    ])
def test_bootstrap_interval(metric, data):
    conf_interval = compute_bootstrap_interval(metric, data[0], data[0])
    assert conf_interval.level == 0.95
    assert conf_interval.limits[0] is not None and conf_interval.limits[1] is not None

    conf_interval = compute_bootstrap_interval(metric, data[0], data[1])
    assert conf_interval.level == 0.95
    assert conf_interval.limits[0] is not None and conf_interval.limits[1] is not None


def test_binomial_interval():
    conf_interval = compute_binomial_interval(pd.Series([1, 1]), pd.Series([0, 0]), confidence_level=0.99)
    assert conf_interval.level == 0.99
    assert conf_interval.limits[0] is not None and conf_interval.limits[1] is not None

    conf_interval = compute_binomial_interval(pd.Series([1, 0]), pd.Series([1, 0]), confidence_level=0.80)
    assert conf_interval.level == 0.80
    assert conf_interval.limits[0] is not None and conf_interval.limits[1] is not None

    conf_interval = compute_binomial_interval(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 1, 0]))
    assert conf_interval.level == 0.95
    assert conf_interval.limits[0] is not None and conf_interval.limits[1] is not None
