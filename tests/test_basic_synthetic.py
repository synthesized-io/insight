import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from synthesized import HighDimSynthesizer
from synthesized.common.values import ContinuousValue, DateValue
from synthesized.metadata import TypeOverride
from synthesized.metadata_new import DataFrameMeta, Float, Integer, MetaExtractor, String
from synthesized.testing.utils import testing_progress_bar


@pytest.mark.slow
def test_continuous_variable_generation():
    r = np.random.normal(loc=5000, scale=1000, size=1000)
    df_original = pd.DataFrame({'r': r})
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=1000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=testing_progress_bar)
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.slow
def test_categorical_similarity_variable_generation():
    r = np.random.normal(loc=10, scale=2, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=1000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=testing_progress_bar)
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.slow
def test_categorical_variable_generation():
    r = np.random.normal(loc=5, scale=1, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=1000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=testing_progress_bar)
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.slow
def test_date_variable_generation():
    df_original = pd.DataFrame({
        'z': pd.date_range(start='01/01/1900', end='01/01/2020', periods=1000).strftime("%d/%m/%Y"),
    }).sample(frac=1.)
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=1000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=testing_progress_bar)

    assert isinstance(synthesizer.df_value['z'], DateValue)


@pytest.mark.slow
def test_nan_producing():
    size = 1000
    nans_prop = 0.33

    df_original = pd.DataFrame({
        'x1': np.random.normal(loc=0, scale=1, size=size),
        'x2': np.random.normal(loc=0, scale=1, size=size),
        'y1': np.random.choice(['A', 'B'], size=size),
        'y2': np.random.choice(['A', 'B'], size=size),
    })
    df_original.loc[np.random.uniform(size=len(df_original)) < nans_prop, 'x2'] = np.nan
    df_original.loc[np.random.uniform(size=len(df_original)) < nans_prop, 'y2'] = np.nan

    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=100, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), produce_nans=True,
                                                progress_callback=testing_progress_bar)
        assert df_synthesized['x1'].isna().sum() == 0
        assert df_synthesized['y1'].isna().sum() == 0
        assert df_synthesized['x2'].isna().sum() > 0
        assert df_synthesized['y2'].isna().sum() > 0

        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), produce_nans=False,
                                                progress_callback=testing_progress_bar)
        assert df_synthesized['x1'].isna().sum() == 0
        assert df_synthesized['y1'].isna().sum() == 0
        assert df_synthesized['x2'].isna().sum() == 0
        assert df_synthesized['y2'].isna().sum() == 0


@pytest.mark.slow
def test_inf_not_producing():
    r = np.random.normal(loc=0, scale=1, size=1000)
    df_original = pd.DataFrame({'r': r}, dtype=np.float32)
    indices = np.random.choice(np.arange(r.size), replace=False, size=int(r.size * 0.1))
    df_original.iloc[indices] = np.inf
    indices = np.random.choice(np.arange(r.size), replace=False, size=int(r.size * 0.1))
    df_original.iloc[indices] = -np.inf
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=1000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=testing_progress_bar)
    assert df_synthesized['r'].isin([np.Inf, -np.Inf]).sum() == 0


@pytest.mark.slow
def test_type_overrides():
    r = np.random.normal(loc=10, scale=2, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    df_meta = MetaExtractor.extract(df=df_original, type_overrides={'r': TypeOverride.CONTINUOUS})
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    assert type(synthesizer.get_values()[0]) == ContinuousValue


@pytest.mark.slow
def test_encode():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n)})
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        _, df_synthesized = synthesizer.encode(df_original)
    assert df_synthesized.shape == df_original.shape


@pytest.mark.slow
def test_encode_unlearned_meta():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n), 'z': np.full(n, 1.0)})

    x = Float('x')
    x.extract(df_original)

    y = String('y')
    y.extract(df_original)

    z = Integer('z')
    z.extract(df_original)

    df_meta = DataFrameMeta(name='df_meta')
    for meta in [x, y, z]:
        df_meta[meta.name] = meta

    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
            synthesizer.learn(num_iterations=10, df_train=df_original)
            _, df_synthesized = synthesizer.encode(df_original)
    assert df_synthesized.shape == df_original.shape


@pytest.mark.slow
def test_encode_deterministic():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n)})
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized = synthesizer.encode_deterministic(df_original)
    assert df_synthesized.shape == df_original.shape
