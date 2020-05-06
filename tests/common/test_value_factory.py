import logging
import warnings

import numpy as np
import pandas as pd
import hypothesis.strategies as st
from hypothesis import given, event
from hypothesis.extra.pandas import column, data_frames, range_indexes

from synthesized.common import ValueFactory
from synthesized.common.values import NanValue, ContinuousValue, CategoricalValue


logger = logging.getLogger()
logger.setLevel(logging.WARNING)
warnings.filterwarnings('ignore', module='sklearn')


@given(df=data_frames(
    [column('A', elements=st.floats(width=32, allow_infinity=False), fill=st.nothing())],
    index=range_indexes(min_size=2, max_size=500)
))
def test_vf_floats(df):
    vf = ValueFactory(df=df)
    value = vf.get_values()[0]
    value_name = ''

    if isinstance(value, NanValue):
        value_name += 'NanValue:'
        value = value.value

    value_name += value.__class__.__name__

    if isinstance(value, ContinuousValue):
        value_name += '(int)' if value.integer else '(float)'

    event(value_name)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


@given(df=data_frames(
    [column('A', elements=st.datetimes(allow_imaginary=False), fill=st.nothing())],
    index=range_indexes(min_size=2, max_size=500)
))
def test_vf_datetimes(df):
    vf = ValueFactory(df=df)
    value = vf.get_values()[0]
    value_name = ''

    if isinstance(value, NanValue):
        value_name += 'NanValue:'
        value = value.value

    value_name += value.__class__.__name__

    if isinstance(value, ContinuousValue):
        value_name += '(int)' if value.integer else '(float)'

    event(value_name)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


@given(
    df=data_frames([column(
        'A',
        elements=st.text(alphabet=st.characters(whitelist_categories=['Lu', 'Nd']), max_size=20),
        fill=st.nothing()
    )], index=range_indexes(min_size=2, max_size=500))
)
def test_vf_text(df):
    vf = ValueFactory(df=df)
    value = vf.get_values()[0]
    value_name = ''

    if isinstance(value, NanValue):
        value_name += 'NanValue:'
        value = value.value

    value_name += value.__class__.__name__

    if isinstance(value, ContinuousValue):
        value_name += '(int)' if value.integer else '(float)'

    event(value_name)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


@given(df=data_frames(
    [column(
        'A',
        elements=st.one_of(st.floats(width=32, allow_infinity=False), st.text(max_size=10)),
        fill=st.nothing()
    )],
    index=range_indexes(min_size=2, max_size=500)
))
def test_vf_text_floats(df):
    vf = ValueFactory(df=df)
    value = vf.get_values()[0]
    value_name = ''

    if isinstance(value, NanValue):
        value_name += 'NanValue:'
        value = value.value

    value_name += value.__class__.__name__

    if isinstance(value, ContinuousValue):
        value_name += '(int)' if value.integer else '(float)'

    event(value_name)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


@given(
    df=data_frames([
        column('A', elements=st.one_of(st.integers(), st.sampled_from([np.NaN, pd.NaT, None])), fill=st.nothing())
    ], index=range_indexes(min_size=2, max_size=500))
)
def test_vf_na_int(df):
    vf = ValueFactory(df=df)
    value = vf.get_values()[0]
    value_name = ''

    if isinstance(value, NanValue):
        value_name += 'NanValue:'
        value = value.value

    value_name += value.__class__.__name__

    if isinstance(value, ContinuousValue):
        value_name += '(int)' if value.integer else '(float)'

    event(value_name)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


# Normal Data Columns
# ----------------------------------------------------------------------
def test_vf_int64():
    df = pd.DataFrame({'int64': list(range(0, 1000))})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


def test_vf_string():
    df = pd.DataFrame({'string': list('abcde')*200})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


def test_vf_uint8():
    df = pd.DataFrame({'uint8': np.arange(0, 1000).astype('u1')})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


def test_vf_float():
    df = pd.DataFrame({'float64': np.arange(0.0, 1000.0)})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


def test_vf_bool():
    df = pd.DataFrame({'bool': [True, False]*500})
    vf = ValueFactory(df=df)

    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


def test_vf_bool_constant():
    df = pd.DataFrame({'bool_false': [False, ]*1000})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


def test_vf_dates():
    df = pd.DataFrame({'dates': pd.date_range('now', periods=1000).values})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


def test_vf_category_string():
    df = pd.DataFrame({'category_string': pd.Categorical(list("ABCDE")*200)})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)


# Missing Value Data Columns
# ----------------------------------------------------------------------

def test_vf_missing_ints():
    df = pd.DataFrame({'missing_ints': np.array([1, 1, 1, 0]*250)/np.array([1, 1, 1, 0]*250) * np.arange(0, 1000)})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)

    value = vf.get_values()[0]
    assert isinstance(value, NanValue)
    assert isinstance(value.value, ContinuousValue)
    assert value.value.integer


def test_vf_missing_strings():
    df = pd.DataFrame({'missing_strings': ['a', 'b', 'c', None]*100})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)

    value = vf.get_values()[0]
    assert isinstance(value, CategoricalValue)
    assert value.categories == ['nan', 'a', 'b', 'c']


def test_vf_missing_categories():
    df = pd.DataFrame({'missing_strings': pd.Categorical(['a', 'b', 1, None]*100)})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)

    value = vf.get_values()[0]
    assert isinstance(value, CategoricalValue)
    assert value.categories == ['nan', '1', 'a', 'b']


def test_vf_double_missing_strings():
    df = pd.DataFrame({'missing_strings': ['a', 'b', np.NaN, None]*100})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)

    value = vf.get_values()[0]
    assert isinstance(value, CategoricalValue)
    assert value.categories == ['nan', 'a', 'b']


def test_vf_double_missing_ints():
    df = pd.DataFrame({'missing_ints': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, np.NaN, None]*2)})

    vf = ValueFactory(df=df)
    df_p = vf.preprocess(df=df)
    vf.postprocess(df=df_p)

    value = vf.get_values()[0]
    assert isinstance(value, NanValue)
