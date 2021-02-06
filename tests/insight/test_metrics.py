import logging

import pandas as pd

from synthesized.insight.metrics import categorical_logistic_correlation, kendell_tau_correlation

logger = logging.getLogger(__name__)


def test_categorical_logistic_correlation_datetimes():
    sr_a = pd.Series([1, 2, 3, 4, 5], name='ints')
    sr_b = pd.to_datetime(
        pd.Series(['10/07/2020', '10/06/2020', '10/12/2020', '1/04/2021', '10/06/2018'], name='dates')
    )

    value = categorical_logistic_correlation(sr_a, sr_b)


def test_kt_correlation_string_numbers():
    sr_a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], name="a")
    sr_b = pd.Series(['1.4', '2.1', '', '4.1', '3.9', '4.4', '5.1', '6.0', '7.5', '9', '11.4', '12.1', '', '14.1', '13.9'], name="b")

    # df['TotalCharges'].dtype is Object in this case, eg. "103.4" instead of 103.4
    kt1 = kendell_tau_correlation(sr_a=sr_a, sr_b=sr_b)

    sr_b = pd.to_numeric(sr_b, errors='coerce')
    kt2 = kendell_tau_correlation(sr_a=sr_a, sr_b=sr_b)

    assert abs(kt1 - kt2) < 0.01
