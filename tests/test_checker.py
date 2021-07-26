import pandas as pd
import numpy as np

import pytest

from ..src.checker import ColumnCheck


@pytest.fixture(scope='module')
def df():
    df = pd.DataFrame({
        'string_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=1000),
        'bool_col': np.random.choice([False, True], size=1000).astype('?'),
        'date_col': pd.to_datetime(18_000 + np.random.normal(500, 50, size=1000).astype(int), unit='D'),
        'int_col': [n for n in [0, 1, 2, 3, 4, 5] for i in range([50, 50, 0, 200, 400, 300][n])],
        'float_col': np.random.normal(0.0, 1.0, size=1000),
        'int_bool_col': np.random.choice([0, 1], size=1000),
        'ordered_cat_col': pd.Categorical(np.random.choice(["b", "d", "c"], size=1000), categories=["b", "c", "d"], ordered=True)
    })

    return df


@pytest.fixture(scope='module')
def df_with_nan(df):
    for col in df.columns:
        df.loc[df.sample(frac=0.1).index, col] = pd.np.nan


def test_columns_types(df, categorical_cols, continuous_cols, date_cols, ordinal_cols):
    check = ColumnCheck()

    pred_categorical_cols = set()
    for col in df.columns:
        if check.categorical(df[col]):
            pred_categorical_cols.add(col)
    assert pred_categorical_cols == categorical_cols

    pred_continuous_cols = set()
    for col in df.columns:
        if check.continuous(df[col]):
            pred_continuous_cols.add(col)
    assert pred_continuous_cols == continuous_cols

    pred_date_cols = set()
    for col in df.columns:
        if check.date(df[col]):
            pred_date_cols.add(col)
    assert pred_date_cols == date_cols

    pred_ordinal_cols = set()
    for col in df.columns:
        if check.ordinal(df[col]):
            pred_ordinal_cols.add(col)
    assert pred_ordinal_cols == ordinal_cols


def test_column_check(df):
    categorical_cols = set(['string_col', 'bool_col', 'int_bool_col', 'ordered_cat_col'])
    continuous_cols = set(['date_col', 'int_col', 'float_col'])
    date_cols = set(['date_col'])
    ordinal_cols = set(['ordered_cat_col'])

    test_columns_types(df, categorical_cols, continuous_cols, date_cols, ordinal_cols)

    # Adding some NaNs to the dataframe
    for col in df.columns:
        df.loc[df.sample(frac=0.1).index, col] = pd.np.nan

    test_columns_types(df, categorical_cols, continuous_cols, date_cols, ordinal_cols)


