import datetime

import numpy as np
import pandas as pd
import hypothesis.strategies as st
from hypothesis import given, event, settings, HealthCheck, seed
from hypothesis.extra.pandas import column, data_frames, range_indexes

from synthesized.metadata import MetaExtractor, ContinuousMeta, CategoricalMeta, AssociationMeta, NanMeta, \
    SamplingMeta, ConstantMeta


@seed(42)
@settings(deadline=None)
@given(df=data_frames(
    [column('A', elements=st.floats(width=32, allow_infinity=False), fill=st.nothing())],
    index=range_indexes(min_size=2, max_size=500)
))
def test_vf_floats(df):
    df_meta = MetaExtractor.extract(df=df)
    value = df_meta.all_values[0]
    value_name = ''

    if isinstance(value, NanMeta):
        value_name += 'NanValue:'
        value = value.value

    value_name += value.__class__.__name__

    if isinstance(value, ContinuousMeta):
        value_name += '(int)' if value.integer else '(float)'

    event(value_name)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


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
    value = df_meta.all_values[0]
    value_name = ''

    if isinstance(value, NanMeta):
        value_name += 'NanMeta:'
        value = value.value

    value_name += value.__class__.__name__

    if isinstance(value, ContinuousMeta):
        value_name += '(int)' if value.integer else '(float)'

    event(value_name)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


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
    value = df_meta.all_values[0]
    value_name = ''

    if isinstance(value, NanMeta):
        value_name += 'NanMeta:'
        value = value.value

    value_name += value.__class__.__name__

    if isinstance(value, ContinuousMeta):
        value_name += '(int)' if value.integer else '(float)'

    event(value_name)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


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
    value = df_meta.all_values[0]
    value_name = ''

    if isinstance(value, NanMeta):
        value_name += 'NanMeta:'
        value = value.value

    value_name += value.__class__.__name__

    if isinstance(value, ContinuousMeta):
        value_name += '(int)' if value.integer else '(float)'

    event(value_name)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


@seed(42)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    df=data_frames([
        column('A', elements=st.one_of(st.integers(), st.sampled_from([np.NaN, pd.NaT, None])), fill=st.nothing())
    ], index=range_indexes(min_size=2, max_size=500))
)
def test_vf_na_int(df):
    df_meta = MetaExtractor.extract(df=df)
    value = df_meta.all_values[0]
    value_name = ''

    if isinstance(value, NanMeta):
        value_name += 'NanMeta:'
        value = value.value

    value_name += value.__class__.__name__

    if isinstance(value, ContinuousMeta):
        value_name += '(int)' if value.integer else '(float)'
        assert value.integer
    elif isinstance(value, SamplingMeta):
        for v in value.categories.index:
            assert v in [pd.NaT, np.NaN] or \
                   sum(df[value.name].isna())/len(df) >= MetaExtractor.parsing_nan_fraction_threshold
    else:
        assert isinstance(value, ConstantMeta) or isinstance(value, CategoricalMeta)

    event(value_name)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


# Normal Data Columns
# ----------------------------------------------------------------------
def test_vf_int64():
    df = pd.DataFrame({'int64': list(range(0, 1000))})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


def test_vf_string():
    df = pd.DataFrame({'string': list('abcde')*200})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


def test_vf_uint8():
    df = pd.DataFrame({'uint8': np.arange(0, 1000).astype('u1')})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


def test_vf_float():
    df = pd.DataFrame({'float64': np.arange(0.0, 1000.0)})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


def test_vf_bool():
    df = pd.DataFrame({'bool': [True, False]*500})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


def test_vf_bool_constant():
    df = pd.DataFrame({'bool_false': [False, ]*1000})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


def test_vf_dates():
    df = pd.DataFrame({'dates': pd.date_range('now', periods=1000).values})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)


def test_vf_category_string():
    df = pd.DataFrame({'category_string': pd.Categorical(list("ABCDE")*200)})

    dp = MetaExtractor.extract(df=df)
    df_p = dp.preprocess(df=df)
    dp.postprocess(df=df_p)


# Missing Value Data Columns
# ----------------------------------------------------------------------

def test_vf_missing_ints():
    df = pd.DataFrame({'missing_ints': np.array([1, 1, 1, 0]*250)/np.array([1, 1, 1, 0]*250) * np.arange(0, 1000)})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)

    value = df_meta.all_values[0]
    assert isinstance(value, NanMeta)
    assert isinstance(value.value, ContinuousMeta)
    assert value.value.integer


def test_vf_missing_strings():
    df = pd.DataFrame({'missing_strings': ['a', 'b', 'c', None]*100})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)

    value = df_meta.all_values[0]
    assert isinstance(value, CategoricalMeta)
    assert value.categories == ['nan', 'a', 'b', 'c']


def test_vf_missing_categories():
    df = pd.DataFrame({'missing_strings': pd.Categorical(['a', 'b', 1, None]*100)})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)

    value = df_meta.all_values[0]
    assert isinstance(value, CategoricalMeta)
    assert value.categories == ['nan', '1', 'a', 'b']


def test_vf_double_missing_strings():
    df = pd.DataFrame({'missing_strings': ['a', 'b', np.NaN, None]*100})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)

    value = df_meta.all_values[0]
    assert isinstance(value, CategoricalMeta)
    assert value.categories == ['nan', 'a', 'b']


def test_vf_double_missing_ints():
    df = pd.DataFrame({'missing_ints': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, np.NaN, None]*2)})

    df_meta = MetaExtractor.extract(df=df)
    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)

    value = df_meta.all_values[0]
    assert isinstance(value, NanMeta)


def test_vf_associated_columns():
    df = pd.DataFrame({
        'car_brand': ['Porsche', 'Volkwagen', 'BMW', 'Porsche', 'Volkwagen', 'BMW']*2,
        'car_model': ['Boxter', 'Polo', 'M3', 'Macaan', 'Golf', 'X5']*2,
        'car_year': [2012, 2016, 2011, 2011, 2014, 2016] + [2012, 2012, 2011, 2013, 2014, 2013]
    })

    associations = {
        'car_brand': ['car_model'],
        'car_model': ['car_year']
    }

    df_meta = MetaExtractor.extract(df=df, associations=associations)
    value = df_meta.association_meta
    assert isinstance(value, AssociationMeta)
    assert value.associations == [['car_brand', 'car_model', 'car_year']]

    df_p = df_meta.preprocess(df=df)
    df_meta.postprocess(df=df_p)
