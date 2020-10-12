import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.metadata import TypeOverride
from synthesized.common.values import ContinuousValue
from synthesized.testing.utils import testing_progress_bar


@pytest.mark.slow
def test_continuous_variable_generation():
    r = np.random.normal(loc=5000, scale=1000, size=1000)
    df_original = pd.DataFrame({'r': r})
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=testing_progress_bar)
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.slow
def test_categorical_similarity_variable_generation():
    r = np.random.normal(loc=10, scale=2, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=testing_progress_bar)
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.slow
def test_categorical_variable_generation():
    r = np.random.normal(loc=5, scale=1, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), progress_callback=testing_progress_bar)
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.slow
def test_nan_producing():
    size = 1000
    x = np.random.normal(loc=0, scale=1, size=size)
    x_nan = np.random.normal(loc=0, scale=1, size=size)
    indices = np.random.choice(np.arange(size), replace=False, size=int(size * 0.33))
    x_nan[indices] = np.nan
    df_original = pd.DataFrame({
        'x': x,
        'x_nan': x_nan,
        'y': np.random.choice(['A', 'B'], size=size),
        'y_nan': np.random.choice(['A', 'B', np.nan], size=size)
    })
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=100, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), produce_nans=True,
                                                progress_callback=testing_progress_bar)
    assert df_synthesized['x_nan'].isna().sum() > 0
    assert df_synthesized['y_nan'].isna().sum() > 0
    assert df_synthesized['x'].isna().sum() == 0
    assert df_synthesized['y'].isna().sum() == 0


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
        synthesizer.learn(num_iterations=2500, df_train=df_original)
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
def test_encode_deterministic():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n)})
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized = synthesizer.encode_deterministic(df_original)
    assert df_synthesized.shape == df_original.shape
