from datetime import date
from typing import Sequence, Type

import hypothesis.strategies as st
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis.extra.pandas import column, columns, data_frames, range_indexes

from synthesized.metadata_new import (Bool, Date, Float, Integer, IntegerBool, MetaExtractor, OrderedString, Ordinal,
                                      String)


@pytest.mark.slow
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(df=data_frames(columns([str(i) for i in range(30)], elements=st.floats(), fill=st.nothing())),
       n_col=st.integers(min_value=0, max_value=30))
def test_data_frame(df, n_col):
    df_meta = MetaExtractor.extract(df=df.iloc[:, :n_col])
    assert len(df_meta) == n_col


@pytest.mark.slow
@given(df=data_frames([column('str', elements=st.text(), fill=st.nothing())],
                      index=range_indexes(min_size=10)))
def test_str(df):
    str_meta = MetaExtractor.extract(df)['str']
    assert isinstance(str_meta, String)


@pytest.mark.slow
@given(df=data_frames([column('o_str', elements=st.floats(), fill=st.nothing())],
                      index=range_indexes(min_size=10)))
def test_ordered_str(df):
    df['o_str'] = df['o_str'].astype(pd.CategoricalDtype(ordered=True))
    str_meta = MetaExtractor.extract(df)['o_str']
    assert isinstance(str_meta, OrderedString)


# helper functions for Ordinal meta testing
def _test_ordinal(meta: Ordinal, OrdinalClass: Type[Ordinal], sr: pd.Series, sort_list: Sequence):
    """boilerplate code for testing Ordinal classes"""
    assert isinstance(meta, OrdinalClass)
    assert meta.min == sr.min()
    assert meta.max == sr.max()
    assert meta.sort(sort_list) == sorted(sort_list)
    assert meta.nan_freq is not None and (meta.nan_freq > 0) == sr.isna().any()


def is_integer_sr(sr: pd.Series):
    """returns true if float elements of pandas series can be represented as integers"""
    return sr.dropna().apply(lambda x: float(x).is_integer()).all()


def is_integer_bool_sr(sr: pd.Series):
    """returns true if elements of pandas series are in 0,1"""
    return sr.dropna().isin([0, 1]).all()


@pytest.mark.slow
@given(df=data_frames([column('float', elements=st.floats(allow_infinity=False), fill=st.nothing())],
                      index=range_indexes(min_size=10)),
       sort_list=st.lists(elements=st.floats(allow_nan=False)))
def test_floats(df, sort_list):
    assume(not is_integer_sr(df['float']))
    float_meta = MetaExtractor.extract(df)['float']
    _test_ordinal(float_meta, Float, df['float'], sort_list)


INT64_RANGE = 9223372036854775807


@pytest.mark.slow
@given(df=data_frames([column('int',
                      elements=st.integers(min_value=-INT64_RANGE, max_value=INT64_RANGE),
                      fill=st.nothing())],
                      index=range_indexes(min_size=10)),
       sort_list=st.lists(elements=st.integers()))
def test_ints(df, sort_list):
    assume(not is_integer_bool_sr(df['int']))
    int_meta = MetaExtractor.extract(df)['int']
    _test_ordinal(int_meta, Integer, df['int'], sort_list)


@pytest.mark.slow
@given(df=data_frames([column('bool_int', elements=st.integers(min_value=0, max_value=1), fill=st.nothing())],
                      index=range_indexes(min_size=10)),
       sort_list=st.lists(elements=st.integers(min_value=0, max_value=1)))
def test_bool_ints(df, sort_list):
    bool_int_meta = MetaExtractor.extract(df)['bool_int']
    _test_ordinal(bool_int_meta, IntegerBool, df['bool_int'], sort_list)


@pytest.mark.slow
@given(df=data_frames([column('bool', elements=st.booleans(), fill=st.nothing())],
                      index=range_indexes(min_size=10)),
       sort_list=st.lists(elements=st.booleans()))
def test_bool(df, sort_list):
    bool_meta = MetaExtractor.extract(df)['bool']
    _test_ordinal(bool_meta, Bool, df['bool'], sort_list)


# possible input date formats
date_formats = (
    '%Y-%m-%d', '%m-%d-%Y', '%d-%m-%Y', '%y-%m-%d', '%m-%d-%y', '%d-%m-%y',
    '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%y/%m/%d', '%m/%d/%y', '%d/%m/%y',
    '%y/%m/%d %H:%M', '%d/%m/%y %H:%M', '%m/%d/%y %H:%M',
    '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%d/%m/%Y %H.%M.%S', '%d/%m/%Y %H:%M:%S'
)
# tuple of sets of formats where each set contains formats that may be confused for one another
ambigious_formats = (
    {'%y-%m-%d', '%m-%d-%y', '%d-%m-%y'},
    {'%y/%m/%d', '%m/%d/%y', '%d/%m/%y'},
    {'%d-%m-%Y', '%m-%d-%Y'},
    {'%Y-%m-%d', '%Y-%d-%m'},
    {'%d/%m/%Y', '%m/%d/%Y'},
    {'%Y/%m/%d', '%Y/%d/%m'},
    {'%y/%m/%d %H:%M', '%d/%m/%y %H:%M', '%m/%d/%y %H:%M'},
)

numpy_date_range = {"min_value": date(1677, 9, 23), "max_value": date(2262, 4, 11)}


@given(df=data_frames([column('date',
                      elements=st.dates(**numpy_date_range), fill=st.nothing())],
                      index=range_indexes(min_size=10)),
       date_format=st.sampled_from(date_formats),
       sort_list=st.lists(elements=st.dates(**numpy_date_range)))
def test_dates(df, date_format, sort_list):
    df["date"] = df["date"].apply(lambda d: d.strftime(date_format))
    date_meta = MetaExtractor.extract(df)['date']

    if date_format != date_meta.date_format:
        # check to see if formats could be confused
        ambigiuty_check = [date_format in formats and date_meta.date_format in formats
                           for formats in ambigious_formats]
        if not any(ambigiuty_check):
            raise AssertionError(f"{date_format} != {date_meta.date_format}")

    df['date'] = pd.to_datetime(df['date'], format=date_format)
    sort_list = pd.to_datetime(sort_list).tolist()
    _test_ordinal(date_meta, Date, df['date'], sort_list)
