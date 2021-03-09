import logging
from datetime import date
from typing import Sequence, Type

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis.extra.pandas import column, columns, data_frames, range_indexes

from synthesized.config import AddressLabels, BankLabels, PersonLabels
from synthesized.metadata_new import (Address, Bank, Bool, DateTime, Float, FormattedString, Integer, IntegerBool,
                                      MetaExtractor, OrderedString, Ordinal, Person, String)

logger = logging.getLogger(__name__)


@pytest.mark.slow
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
    assert meta.min == np.nanmin(sr.values.astype(meta.dtype))
    assert meta.max == np.nanmax(sr.values.astype(meta.dtype))
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

        # Given that the date format is ambigious, make it the same as the meta's date format.
        date_format = date_meta.date_format

    df['date'] = pd.to_datetime(df['date'], format=date_format)
    sort_list = pd.to_datetime(sort_list).tolist()
    _test_ordinal(date_meta, DateTime, df['date'], sort_list)


def test_annotations():

    df = pd.DataFrame({
        'a': ['a', 'b', 'c'],
        'b': ['MAUS', 'HBUK', 'HBUK'],
        'c': ['010468', '616232', '131315'],
        'd': ['d', 'm', 'm'],
        'e': ['Alice', 'Bob', 'Charlie'],
        'f': ['Smith', 'Holmes', 'Smith'],
        'g': ['SJ-3921', 'LE-0826', 'PQ-0871'],
    })

    annotations = [
        Address(name='address', labels=AddressLabels(city_label='a', street_label='d')),
        Bank(name='bank', labels=BankLabels(bic_label='b', sort_code_label='c')),
        Person(name='person', labels=PersonLabels(firstname_label='e', lastname_label='f')),
        FormattedString(name='g', pattern='[A-Z]{2}-[0-9]{4}'),
    ]

    df_meta = MetaExtractor.extract(df=df, annotations=annotations)

    assert sorted(list(df_meta['address'].keys())) == ['a', 'd']
    assert sorted(list(df_meta['bank'].keys())) == ['b', 'c']
    assert sorted(list(df_meta['person'].keys())) == ['e', 'f']
    assert sorted(list(df_meta['g'].keys())) == []


def test_associations():

    df = pd.DataFrame({
        'a': [0, 0, 1, 1],
        'b': [0, 1, 0, 1],
        'c': [0, 0, 0, 1]
    })

    df_meta = MetaExtractor.extract(df, associations=[['a', 'b', 'c']])

    true_binding_mask = np.array([[[1, 0], [1, 0]], [[1, 0], [0, 1]]])
    np.testing.assert_array_almost_equal(df_meta["association_a_b_c"].binding_mask, true_binding_mask)
