import datetime

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, event, example, given, seed, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes

from synthesized.config import MetaExtractorConfig
from synthesized.metadata_new.base import Affine, Ring
from synthesized.metadata_new.factory import MetaExtractor
from synthesized.metadata_new.value import DateTime, Float, Integer, String


@pytest.mark.slow
def test_pre_post_processing():
    df = pd.read_csv('data/unittest.csv')
    df_meta = MetaExtractor.extract(df=df)


@pytest.mark.slow
@seed(42)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(df=data_frames(
    [column('A', elements=st.floats(width=32, allow_infinity=False), fill=st.nothing())],
    index=range_indexes(min_size=2, max_size=500)
))
def test_vf_floats(df):
    df_meta = MetaExtractor.extract(df=df)
    value = df_meta['A']
    value_name = value.__class__.__name__
    event(value_name)
    assert isinstance(value, Ring)

@pytest.mark.slow
@seed(42)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(df=data_frames(
    [column('A', elements=st.floats(width=32, allow_infinity=True), fill=st.nothing())],
    index=range_indexes(min_size=2, max_size=500)
))
def test_vf_floats_inf(df):
    df_meta = MetaExtractor.extract(df=df)
    value = df_meta['A']
    value_name = value.__class__.__name__
    event(value_name)
    assert isinstance(value, Ring)


@pytest.mark.slow
@seed(42)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(df=data_frames(
    [column(
        'A',
        elements=st.datetimes(
            min_value=datetime.datetime(2000, 1, 1),
            max_value=datetime.datetime(2020, 3, 1),
            allow_imaginary=False,
            timezones=st.just(datetime.timezone.utc)
        ),
        fill=st.nothing())],
    index=range_indexes(min_size=2, max_size=500)
))
def test_vf_datetimes(df):
    df_meta = MetaExtractor.extract(df=df)
    value = df_meta['A']
    value_name = value.__class__.__name__
    event(value_name)
    assert isinstance(value, DateTime)


@pytest.mark.slow
@seed(42)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    df=data_frames([column(
        'A',
        elements=st.text(alphabet=st.characters(whitelist_categories=['Lu', 'Nd']), max_size=15),
        fill=st.nothing()
    )], index=range_indexes(min_size=2, max_size=500))
)
def test_vf_text(df):
    df_meta = MetaExtractor.extract(df=df)
    value = df_meta['A']
    value_name = value.__class__.__name__
    event(value_name)
    assert isinstance(value, String)


@pytest.mark.slow
@seed(42)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(df=data_frames(
    [column(
        'A',
        elements=st.one_of(st.floats(width=32, allow_infinity=False),
                           st.text(alphabet=st.characters(whitelist_categories=['Lu', 'Nd']), max_size=10)),
        fill=st.nothing()
    )],
    index=range_indexes(min_size=2, max_size=500)
))
def test_vf_text_floats(df):
    df_meta = MetaExtractor.extract(df=df)
    value = df_meta['A']
    value_name = value.__class__.__name__
    event(value_name)
    assert isinstance(value, Ring)


@pytest.mark.slow
@seed(42)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    df=data_frames([
        column('A', elements=st.one_of(st.integers(), st.sampled_from([np.NaN, pd.NaT, None])), fill=st.nothing())
    ], index=range_indexes(min_size=2, max_size=500))
)
@example(df=pd.DataFrame(data=[np.NaN, pd.NaT], index=[0, 1], columns=['A'], dtype=object))
def test_vf_na_int(df):
    print(df)
    df_meta = MetaExtractor.extract(df=df)
    value = df_meta['A']
    value_name = value.__class__.__name__
    event(value_name)


# Normal Data Columns
# ----------------------------------------------------------------------
def test_vf_int64():
    df = pd.DataFrame({'int64': list(range(0, 1000))})
    df_meta = MetaExtractor.extract(df=df)
    assert isinstance(df_meta['int64'], Integer)

def test_vf_string():
    df = pd.DataFrame({'string': list('abcde')*200})
    df_meta = MetaExtractor.extract(df=df)

def test_vf_uint8():
    df = pd.DataFrame({'uint8': np.arange(0, 1000).astype('u1')})
    df_meta = MetaExtractor.extract(df=df)


def test_vf_float():
    df = pd.DataFrame({'float64': np.arange(0.0, 1000.0)})
    df_meta = MetaExtractor.extract(df=df)


def test_vf_bool():
    df = pd.DataFrame({'bool': [True, False]*500})
    df_meta = MetaExtractor.extract(df=df)


def test_vf_bool_constant():
    df = pd.DataFrame({'bool_false': [False, ]*1000})
    df_meta = MetaExtractor.extract(df=df)


def test_vf_dates():
    df = pd.DataFrame({'dates': pd.date_range('now', periods=1000).values})
    df_meta = MetaExtractor.extract(df=df)


def test_vf_category_string():
    df = pd.DataFrame({'category_string': pd.Categorical(list("ABCDE")*200)})
    dp = MetaExtractor.extract(df=df)


# Missing Value Data Columns
# ----------------------------------------------------------------------
def test_vf_missing_ints():
    df = pd.DataFrame({'missing_ints': np.array([1, 1, 1, 0]*250)/np.array([1, 1, 1, 0]*250) * np.arange(0, 1000)})
    df_meta = MetaExtractor.extract(df=df)

    value = df_meta['missing_ints']
    assert isinstance(value, Integer)


def test_vf_missing_strings():
    df = pd.DataFrame({'missing_strings': ['a', 'b', 'c', None]*100})

    df_meta = MetaExtractor.extract(df=df)

    value = df_meta['missing_strings']
    assert isinstance(value, String)
    assert value.categories == ['a', 'b', 'c']
    assert value.nan_freq > 0


def test_vf_missing_categories():
    df = pd.DataFrame({'missing_strings': pd.Categorical(['a', 'b', 1, None]*100)})

    df_meta = MetaExtractor.extract(df=df)

    value = df_meta['missing_strings']
    assert isinstance(value, String)
    assert value.categories == ['a', 'b', '1']
    assert value.nan_freq > 0


def test_vf_double_missing_strings():
    df = pd.DataFrame({'missing_strings': ['a', 'b', np.NaN, None]*100})

    df_meta = MetaExtractor.extract(df=df)

    value = df_meta['missing_strings']
    assert isinstance(value, String)
    assert value.categories == ['a', 'b']
    assert value.nan_freq > 0


def test_vf_double_missing_ints():
    df = pd.DataFrame({'missing_ints': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, np.NaN, None]*2)})

    df_meta = MetaExtractor.extract(df=df)

    value = df_meta['missing_ints']
    assert isinstance(value, Integer)
    assert value.nan_freq > 0
