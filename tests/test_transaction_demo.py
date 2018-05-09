import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from examples.transaction_demo import TransactionVectorizer


def test_TransactionTransformer():
    df = pd.DataFrame.from_records([
        {'account_id': 'aaa', 'day': 1, 'operation': 1, 'amount': 100.0},
        {'account_id': 'bbb', 'day': 2, 'operation': 2, 'amount': 200.0},
    ])
    trans = TransactionVectorizer(group_column='account_id',
                                  value_column='amount',
                                  dim1_column='operation',
                                  dim2_column='day',
                                  dim1_size=5,
                                  dim2_size=32)
    df_transformed = trans.fit_transform(df)

    a1 = np.zeros(5 * 32, dtype=np.float64)
    a1[1 * 5 + 1] = 100.0

    a2 = np.zeros(5 * 32, dtype=np.float64)
    a2[2 * 5 + 2] = 200.0

    df_expected = pd.DataFrame.from_records([a1, a2])

    assert_frame_equal(df_transformed, df_expected)

    df_transformed_back = trans.inverse_transform(df_transformed)

    assert_series_equal(df['amount'], df_transformed_back['amount'])
    assert_series_equal(df['day'], df_transformed_back['day'])
    assert_series_equal(df['operation'], df_transformed_back['operation'])
