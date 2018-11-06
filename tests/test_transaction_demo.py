import pandas as pd
from pandas.testing import assert_series_equal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from examples.old_notebooks.transaction_demo import TransactionVectorizer


def test_pipeline():
    transform_pipeline = Pipeline([
        ("vectorizer", TransactionVectorizer(
            group_column='account_id',
            value_column='amount',
            dim1_column='operation',
            dim2_column='day',
            dim1_size=5,
            dim2_size=32)),
        ("scale", StandardScaler())
    ])

    df = pd.DataFrame.from_records([
        {'account_id': 'aaa', 'day': 1, 'operation': 1, 'amount': 100.0},
        {'account_id': 'aaa', 'day': 2, 'operation': 3, 'amount': 300.0},
        {'account_id': 'bbb', 'day': 2, 'operation': 2, 'amount': 200.0},
    ])

    df_trans = transform_pipeline.fit_transform(df)
    df_transformed_back = transform_pipeline.inverse_transform(df_trans)
    assert_series_equal(df['amount'], df_transformed_back['amount'])
    assert_series_equal(df['day'], df_transformed_back['day'])
    assert_series_equal(df['operation'], df_transformed_back['operation'])
