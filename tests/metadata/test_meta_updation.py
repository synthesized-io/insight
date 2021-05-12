import numpy as np
import pandas as pd
from synthesized.metadata.factory import MetaExtractor


def test_nominal_update():
    df = pd.DataFrame(pd.Series(['a', 'a', np.nan, 'b', 'c', np.nan, 'b', 'd'], dtype=object, name='str_col'))
    str_meta = MetaExtractor.extract(df)['str_col']
    assert str_meta.num_rows == 8
    assert set(str_meta.categories) == set(['a', 'b', 'c', 'd'])
    assert str_meta.nan_freq == 0.25

    new_df = pd.DataFrame(pd.Series(['a', 'e', 'f', 'b', 'b', np.nan, 'f'], dtype=object, name='str_col'))
    str_meta.update_meta(new_df)
    assert str_meta.num_rows == 15
    assert set(str_meta.categories) == set(['a', 'b', 'c', 'd', 'e', 'f'])
    assert str_meta.nan_freq == 0.2

    new_df = pd.DataFrame(pd.Series(['g', 'h', 't', 't', 't'], dtype=object, name='str_col'))
    str_meta.update_meta(new_df)
    assert str_meta.num_rows == 20
    assert set(str_meta.categories) == set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 't'])
    assert str_meta.nan_freq == 0.15


def test_ordered_string_update():
    df = pd.DataFrame({'ostr_col': pd.Categorical(['plum', 'peach', 'pair', 'each', np.nan, 'plum', 'plum', 'each'],
                       dtype=pd.CategoricalDtype(['each', 'peach', 'pair', 'plum'], ordered=True))})

    ostr_meta = MetaExtractor.extract(df)['ostr_col']
    assert ostr_meta.num_rows == 8
    assert ostr_meta.categories == ['each', 'peach', 'pair', 'plum']
    assert ostr_meta.nan_freq == 0.125

    new_df = pd.DataFrame({'ostr_col': pd.Categorical(['plum', 'peach', 'apple', 'pear', np.nan, np.nan, 'guava'],
                           dtype=pd.CategoricalDtype(['apple', 'peach', 'pear', 'plum', 'guava'], ordered=True))})

    ostr_meta.update_meta(new_df)
    assert ostr_meta.num_rows == 15
    assert ostr_meta.categories == ['apple', 'each', 'peach', 'pear', 'pair', 'plum', 'guava']
    assert ostr_meta.nan_freq == 0.2

    new_df = pd.DataFrame({'ostr_col': pd.Categorical([np.nan, np.nan, 'each', 'pear', 'guava'],
                           dtype=pd.CategoricalDtype(['each', 'pear', 'guava'], ordered=True))})

    ostr_meta.update_meta(new_df)
    assert ostr_meta.num_rows == 20
    assert ostr_meta.categories == ['apple', 'each', 'peach', 'pear', 'pair', 'plum', 'guava']
    assert ostr_meta.nan_freq == 0.25


def test_ordinal_bool_update():
    df = pd.DataFrame(pd.Series([True, True, True, True], dtype=bool, name='bool_col'))
    bool_meta = MetaExtractor.extract(df)['bool_col']
    assert bool_meta.num_rows == 4
    assert set(bool_meta.categories) == set([True])
    assert bool_meta.min == True
    assert bool_meta.max == True

    new_df = pd.DataFrame(pd.Series([False, False, True, True, False], dtype=bool, name='bool_col'))
    bool_meta.update_meta(new_df)
    assert set(bool_meta.categories) == set([True, False])
    assert bool_meta.num_rows == 9
    assert bool_meta.min == False
    assert bool_meta.max == True


def test_ordinal_int_bool_update():
    df = pd.DataFrame(pd.Series([0, 0, 0, 0], dtype='int64', name='int_bool_col'))
    int_bool_meta = MetaExtractor.extract(df)['int_bool_col']
    assert int_bool_meta.num_rows == 4
    assert set(int_bool_meta.categories) == set([0])
    assert int_bool_meta.min == 0
    assert int_bool_meta.max == 0

    new_df = pd.DataFrame(pd.Series([1, 0, 0, 0, 0], dtype='int64', name='int_bool_col'))
    int_bool_meta.update_meta(new_df)
    assert set(int_bool_meta.categories) == set([1, 0])
    assert int_bool_meta.num_rows == 9
    assert int_bool_meta.min == 0
    assert int_bool_meta.max == 1


def test_ordinal_int_update():
    df = pd.DataFrame(pd.Series([0, -3, 2, 7, 1, 4], dtype='int64', name='int_col'))
    int_meta = MetaExtractor.extract(df)['int_col']
    assert int_meta.num_rows == 6
    assert int_meta.min == -3
    assert int_meta.max == 7

    new_df = pd.DataFrame(pd.Series([-100, 13, -22, 3, -41, 4, 0, 1], dtype='int64', name='int_col'))
    int_meta.update_meta(new_df)
    assert int_meta.num_rows == 14
    assert int_meta.min == -100
    assert int_meta.max == 13

    new_df = pd.DataFrame(pd.Series([-90, 11, 12], dtype='int64', name='int_col'))
    int_meta.update_meta(new_df)
    assert int_meta.num_rows == 17
    assert int_meta.min == -100
    assert int_meta.max == 13


def test_ordinal_float_update():
    df = pd.DataFrame(pd.Series([-3.3, 1.0, 5, 1.0, 13], dtype='float64', name='float_col'))
    float_meta = MetaExtractor.extract(df)['float_col']
    assert float_meta.num_rows == 5
    assert float_meta.min == -3.3
    assert float_meta.max == 13

    new_df = pd.DataFrame(pd.Series([-10, -4, 0, 0, 6.8], dtype='float64', name='float_col'))
    float_meta.update_meta(new_df)
    assert float_meta.num_rows == 10
    assert float_meta.min == -10
    assert float_meta.max == 13


def test_datetime_update():
    df = pd.DataFrame(pd.Series(['2021-01-16', '2021-05-23', '2020-03-01', '2021-02-22'], name='datetime_col', dtype='datetime64[ns]'))
    datetime_meta = MetaExtractor.extract(df)['datetime_col']
    assert datetime_meta.num_rows == 4
    assert datetime_meta.min == np.datetime64('2020-03-01')
    assert datetime_meta.max == np.datetime64('2021-05-23')
    assert datetime_meta.nan_freq == 0

    new_df = pd.DataFrame(pd.Series([np.nan, '2019-01-16', np.nan, '2020-01-16'], dtype='datetime64[ns]', name='datetime_col'))
    datetime_meta.update_meta(new_df)
    assert datetime_meta.num_rows == 8
    assert datetime_meta.min == np.datetime64('2019-01-16')
    assert datetime_meta.max == np.datetime64('2021-05-23')
    assert datetime_meta.nan_freq == 0.25


def test_dataframe_update():
    df = pd.DataFrame({'x0': pd.Series(['a', 'a', 'b', 'a'], dtype=object),
                       'x1': pd.Series([-2, 9, -12, 15], dtype='int64'),
                       'x2': pd.Series([-11.2, -9.3, -8.1, -7], dtype='float64'),
                       'x3': pd.Series([-9, 12, 9.8, -102.98], dtype='float64'),
                       'x4': pd.Series([1, 1, 1, 1], dtype='int64'),
                       'x5': pd.Categorical(['carriage', 'car', np.nan, 'cart'],
                                            dtype=pd.CategoricalDtype(['cart', 'carriage', 'car'], ordered=True)),
                       'x6': pd.Series(['2021-01-25', '2021-05-31', '2020-12-12', np.nan],
                                       name='datetime_col', dtype='datetime64[ns]')})

    df_meta = MetaExtractor.extract(df)
    assert df_meta.num_rows == 4
    assert set(df_meta['x0'].categories) == set(['a', 'b'])
    assert df_meta['x1'].min == -12
    assert df_meta['x1'].max == 15
    assert df_meta['x2'].min == -11.2
    assert df_meta['x2'].max == -7
    assert df_meta['x3'].min == -102.98
    assert df_meta['x3'].max == 12
    assert df_meta['x4'].min == 1
    assert df_meta['x4'].max == 1
    assert df_meta['x5'].categories == ['cart', 'carriage', 'car']
    assert df_meta['x5'].nan_freq == 0.25
    assert df_meta['x6'].min == np.datetime64('2020-12-12')
    assert df_meta['x6'].max == np.datetime64('2021-05-31')
    assert df_meta['x6'].nan_freq == 0.25

    new_df = pd.DataFrame({'x0': pd.Series(['c', 'd', 'a'], dtype=object),
                           'x1': pd.Series([-21, 90, -120], dtype='int64'),
                           'x2': pd.Series([1.1, 1.2, 1.3], dtype='float64'),
                           'x3': pd.Series([-2.5, 9.5, -0.5], dtype='float64'),
                           'x4': pd.Series([0, 0, 0], dtype='int64'),
                           'x5': pd.Categorical(['car', 'horse', 'bus'],
                                                dtype=pd.CategoricalDtype(['horse', 'car', 'bus'], ordered=True)),
                           'x6': pd.Series(['2011-01-16', np.nan, np.nan],
                                           name='datetime_col', dtype='datetime64[ns]')})

    df_meta.update_meta(new_df)
    assert df_meta.num_rows == 7
    assert set(df_meta['x0'].categories) == set(['a', 'b', 'c', 'd'])
    assert df_meta['x1'].min == -120
    assert df_meta['x1'].max == 90
    assert df_meta['x2'].min == -11.2
    assert df_meta['x2'].max == 1.3
    assert df_meta['x3'].min == -102.98
    assert df_meta['x3'].max == 12
    assert df_meta['x4'].min == 0
    assert df_meta['x4'].max == 1
    assert df_meta['x5'].categories == ['horse', 'cart', 'carriage', 'car', 'bus']
    assert df_meta['x5'].nan_freq == 0.14285714285714285
    assert df_meta['x6'].min == np.datetime64('2011-01-16')
    assert df_meta['x6'].max == np.datetime64('2021-05-31')
    print('nan freq: ', df_meta['x6'])
    assert df_meta['x6'].nan_freq == 0.42857142857142855

    new_df = pd.DataFrame({'x0': pd.Series(['c'], dtype=object),
                           'x1': pd.Series([1000], dtype='int64'),
                           'x2': pd.Series([1.1], dtype='float64'),
                           'x3': pd.Series([-250], dtype='float64'),
                           'x4': pd.Series([1], dtype='int64'),
                           'x5': pd.Categorical([np.nan],
                                                dtype=pd.CategoricalDtype([], ordered=True)),
                           'x6': pd.Series(['2022-01-16'],
                                           name='datetime_col', dtype='datetime64[ns]')})

    df_meta.update_meta(new_df)
    assert df_meta.num_rows == 8
    assert set(df_meta['x0'].categories) == set(['a', 'b', 'c', 'd'])
    assert df_meta['x1'].min == -120
    assert df_meta['x1'].max == 1000
    assert df_meta['x2'].min == -11.2
    assert df_meta['x2'].max == 1.3
    assert df_meta['x3'].min == -250
    assert df_meta['x3'].max == 12
    assert df_meta['x4'].min == 0
    assert df_meta['x4'].max == 1
    assert df_meta['x5'].categories == ['horse', 'cart', 'carriage', 'car', 'bus']
    assert df_meta['x5'].nan_freq == 0.25
    assert df_meta['x6'].min == np.datetime64('2011-01-16')
    assert df_meta['x6'].max == np.datetime64('2022-01-16')
    assert df_meta['x6'].nan_freq == 0.375
