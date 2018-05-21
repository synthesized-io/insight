import pandas as pd
from nose.tools import assert_equals
from pandas.util.testing import assert_frame_equal

from synthesized.testing.linkage_attack import linkage_attack
from synthesized.testing.linkage_attack import t_closeness_check, find_neighbour_distances, \
    find_eq_class_fuzzy, find_eq_class


def test_t_closeness_check():
    df = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 1, 'attr2': 1},
        {'key1': 2, 'key2': 3, 'attr1': 2, 'attr2': 1}
    ])
    assert_equals(len(t_closeness_check(df, threshold=0.2)), 24)

    df = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 1, 'attr1': 1, 'attr2': 1},
        {'key1': 1, 'key2': 1, 'attr1': 1, 'attr2': 1}
    ])
    assert_equals(len(t_closeness_check(df, threshold=0.2)), 0)


def test_find_boundaries():
    df = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 3, 'attr2': 4},
        {'key1': 5, 'key2': 6, 'attr1': 7, 'attr2': 8},
        {'key1': 10, 'key2': 11, 'attr1': 12, 'attr2': 13},
    ])

    down, up = find_neighbour_distances(df, {'key1': 10, 'key2': 11}, {})
    assert_equals(down, {'key1': 5., 'key2': 5.})
    assert_equals(up, {})

    down, up = find_neighbour_distances(df, {'key1': 1, 'key2': 2}, {})
    assert_equals(down, {})
    assert_equals(up, {'key1': 4., 'key2': 4.})

    down, up = find_neighbour_distances(df, {'key1': 5, 'key2': 6}, {})
    assert_equals(down, {'key1': 4., 'key2': 4.})
    assert_equals(up, {'key1': 5., 'key2': 5.})


def test_find_eq_class():
    df = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 'a', 'attr1': 3, 'attr2': 4},
        {'key1': 1, 'key2': 'a', 'attr1': 7, 'attr2': 8},
        {'key1': 10, 'key2': 'b', 'attr1': 12, 'attr2': 13},
    ])

    df_expected = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 'a', 'attr1': 3, 'attr2': 4},
        {'key1': 1, 'key2': 'a', 'attr1': 7, 'attr2': 8},
    ])

    found = find_eq_class(df, {'key1': 1, 'key2': 'a'}).reset_index(drop=True)
    assert_frame_equal(df_expected, found)


def test_find_eq_class_fuzzy():
    df1 = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 'a', 'attr1': 3, 'attr2': 4},
        {'key1': 5, 'key2': 'a', 'attr1': 7, 'attr2': 8},
        {'key1': 10, 'key2': 'a', 'attr1': 12, 'attr2': 13},
    ])

    df2 = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 'a', 'attr1': 3, 'attr2': 4},
        {'key1': 5.5, 'key2': 'a', 'attr1': 7.5, 'attr2': 8.5},
        {'key1': 9, 'key2': 'a', 'attr1': 11, 'attr2': 12},
    ])

    df_expected = pd.DataFrame.from_records([
        {'key1': 5.5, 'key2': 'a', 'attr1': 7.5, 'attr2': 8.5},
    ])

    attrs = {'key1': 5, 'key2': 'a'}
    down, up = find_neighbour_distances(df1, attrs, {'key2'})
    found = find_eq_class_fuzzy(df2, attrs, down, up, categ_columns={'key2'}).reset_index(drop=True)

    assert_frame_equal(df_expected, found)


def test_linkage_attack():
    df1 = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 3, 'attr2': 4},
        {'key1': 1, 'key2': 6, 'attr1': 7, 'attr2': 8},
        {'key1': 10, 'key2': 11, 'attr1': 12, 'attr2': 13},
    ])

    df2 = pd.DataFrame.from_records([
        {'key1': 1, 'key2': 2, 'attr1': 3, 'attr2': 4},
        {'key1': 1, 'key2': 6, 'attr1': 7, 'attr2': 8},
        {'key1': 10, 'key2': 11, 'attr1': 12, 'attr2': 13},
    ])

    assert_equals(42, len(linkage_attack(df1, df2, {})))
