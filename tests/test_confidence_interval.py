import pandas as pd
import pytest

from src.synthesized_insight.metrics import BinomialDistanceTest, EarthMoversDistance, HellingerDistance
from src.synthesized_insight.metrics.confidence_interval import BinomialInterval, BootstrapConfidenceInterval


@pytest.mark.parametrize(
    'metric, data', [
        (EarthMoversDistance(), (pd.Series(['a', 'b', 'c']), pd.Series(['c', 'b', 'a']))),
        (HellingerDistance(), (pd.Series([1, 2, 3]), pd.Series([0, 0, 0])))
    ])
def test_bootstrap_interval(metric, data):

    bootstrap_interval = BootstrapConfidenceInterval(metric_cls_obj=metric)
    conf_interval = bootstrap_interval(data[0], data[0])
    assert conf_interval.level == bootstrap_interval.confidence_level
    assert conf_interval.limits[0] is not None and conf_interval.limits[1] is not None

    conf_interval = bootstrap_interval(data[0], data[1])
    assert conf_interval.level == bootstrap_interval.confidence_level
    assert conf_interval.limits[0] is not None and conf_interval.limits[1] is not None


def test_binomial_interval():
    bin_interval = BinomialInterval(metric_cls_obj=BinomialDistanceTest(), confidence_level=0.99)
    conf_interval = bin_interval(pd.Series([1, 1]), pd.Series([0, 0]))
    assert conf_interval.level == 0.99
    assert conf_interval.limits[0] is not None and conf_interval.limits[1] is not None

    bin_interval = BinomialInterval(metric_cls_obj=BinomialDistanceTest(), confidence_level=0.80)
    conf_interval = bin_interval(pd.Series([1, 0]), pd.Series([1, 0]))
    assert conf_interval.level == 0.80
    assert conf_interval.limits[0] is not None and conf_interval.limits[1] is not None

    bin_interval = BinomialInterval(metric_cls_obj=BinomialDistanceTest())
    conf_interval = bin_interval(pd.Series([1, 0, 1, 1]), pd.Series([1, 0, 1, 0]))
    assert conf_interval.level == 0.95
    assert conf_interval.limits[0] is not None and conf_interval.limits[1] is not None
