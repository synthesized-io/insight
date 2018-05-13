import pandas as pd
from nose.tools import assert_equals

from synthesized.testing.t_closeness import max_emd, t_closeness_check


def test_max_emd_should_be_zero_for_same_dfs():
    df1 = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 1},
        {'key1': 2, 'key2': 3, 'attr1': 2}
    ])
    df2 = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 1},
        {'key1': 2, 'key2': 3, 'attr1': 2}
    ])
    column, distance = max_emd(df1, df2, ['attr1'])
    assert_equals(column, 'attr1')
    assert_equals(distance, 0.0)


def test_max_emd_should_be_grater_than_zero():
    df1 = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 1},
        {'key1': 2, 'key2': 3, 'attr1': 2}
    ])
    df2 = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 1},
        {'key1': 2, 'key2': 3, 'attr1': 4}
    ])
    column, distance = max_emd(df1, df2, ['attr1'])
    assert_equals(column, 'attr1')
    assert_equals(distance, 0.5)


def test_max_emd_should_be_greatest():
    df1 = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 1, 'attr2': 1},
        {'key1': 2, 'key2': 3, 'attr1': 2, 'attr2': 1}
    ])
    df2 = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 1, 'attr2': 1},
        {'key1': 2, 'key2': 3, 'attr1': 4, 'attr2': 1}
    ])
    column, distance = max_emd(df1, df2, ['attr1', 'attr2'])
    assert_equals(column, 'attr1')
    assert_equals(distance, 0.5)


def test_t_closeness_check():
    df = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 1, 'attr2': 1},
        {'key1': 2, 'key2': 3, 'attr1': 2, 'attr2': 1}
    ])
    assert_equals(len(t_closeness_check(df, threshold=0.2)), 12)

    df = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 1, 'attr1': 1, 'attr2': 1},
        {'key1': 1, 'key2': 1, 'attr1': 1, 'attr2': 1}
    ])
    assert_equals(len(t_closeness_check(df, threshold=0.2)), 0)
