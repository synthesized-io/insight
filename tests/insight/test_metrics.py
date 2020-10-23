import logging

import pandas as pd
import pytest

from synthesized.insight.metrics import categorical_logistic_correlation
logger = logging.getLogger(__name__)


@pytest.mark.fast
def test_categorical_logistic_correlation_datetimes():
    sr_a = pd.Series([1, 2, 3, 4, 5], name='ints')
    sr_b = pd.to_datetime(
        pd.Series(['10/07/2020', '10/06/2020', '10/12/2020', '1/04/2021', '10/06/2018'], name='dates')
    )

    value = categorical_logistic_correlation(sr_a, sr_b)
