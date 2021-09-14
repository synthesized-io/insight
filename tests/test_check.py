import copy

import numpy as np
import pandas as pd
import pytest

from synthesized_insight import ColumnCheck


@pytest.fixture(scope='module')
def df():
    df = pd.DataFrame({
        'string_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=1000),
        'bool_col': np.random.choice([False, True], size=1000).astype('?'),
        'date_col': pd.to_datetime(18_000 + np.random.normal(500, 50, size=1000).astype(int), unit='D'),
        'int_col': np.random.randint(10, 100000, 1000),
        'float_col': np.random.normal(0.0, 1.0, size=1000),
        'int_bool_col': np.random.choice([0, 1], size=1000),
        'ordered_cat_col': pd.Categorical(np.random.choice(["b", "d", "c"], size=1000), categories=["b", "c", "d"], ordered=True)
    })

    return df


def verify_columns_types(df, categorical_cols, continuous_cols, affine_cols, ordinal_cols):
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

    pred_affine_cols = set()
    for col in df.columns:
        if check.affine(df[col]):
            pred_affine_cols.add(col)
    assert pred_affine_cols == affine_cols

    pred_ordinal_cols = set()
    for col in df.columns:
        if check.ordinal(df[col]):
            pred_ordinal_cols.add(col)
    assert pred_ordinal_cols == ordinal_cols


def test_column_check(df):
    categorical_cols = set(['string_col', 'bool_col', 'int_bool_col', 'ordered_cat_col'])
    continuous_cols = set(['int_col', 'float_col'])
    affine_cols = set(['date_col', 'int_col', 'float_col'])
    ordinal_cols = set(['ordered_cat_col', 'date_col', 'float_col', 'int_col', 'int_bool_col', 'bool_col'])

    verify_columns_types(df.copy(), categorical_cols, continuous_cols, affine_cols, ordinal_cols)

    # Adding some NaNs to the dataframe
    df_nan = copy.deepcopy(df)
    for col in df_nan.columns:
        df_nan.loc[df_nan.sample(frac=0.1).index, col] = np.nan

    verify_columns_types(df_nan, categorical_cols, continuous_cols, affine_cols, ordinal_cols)


def test_check_ordinal():
    check = ColumnCheck()

    sr = pd.Series([1, 2, 3, 4, 5])
    assert check.ordinal(sr) is True

    sr = pd.Series([3 for _ in range(100)] + [4 for _ in range(100)])
    assert check.ordinal(sr) is True

    sr = pd.Series([3 for _ in range(100)])
    assert check.ordinal(sr) is True

    sr = pd.Series(pd.Categorical(np.random.choice(["t2", "t1", "t0"], size=100), categories=["t0", "t1", "t2"], ordered=True))
    assert check.ordinal(sr) is True
