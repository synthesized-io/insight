import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.complex.highdim import HighDimConfig
from synthesized.common.values import TypeOverride, ContinuousValue


@pytest.mark.integration
def test_continuous_variable_generation():
    r = np.random.normal(loc=5000, scale=1000, size=1000)
    df_original = pd.DataFrame({'r': r})
    dp = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(data_panel=dp) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.integration
def test_categorical_similarity_variable_generation():
    r = np.random.normal(loc=10, scale=2, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    dp = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(data_panel=dp) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.integration
def test_categorical_variable_generation():
    r = np.random.normal(loc=5, scale=1, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    dp = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(data_panel=dp) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.integration
def test_nan_producing():
    r = np.random.normal(loc=0, scale=1, size=1000)
    indices = np.random.choice(np.arange(r.size), replace=False, size=int(r.size * 0.2))
    r[indices] = np.nan
    df_original = pd.DataFrame({'r': r})
    dp = MetaExtractor.extract(df=df_original)
    config = HighDimConfig(produce_nans=True)
    with HighDimSynthesizer(data_panel=dp, config=config) as synthesizer:
        synthesizer.learn(num_iterations=2500, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    assert df_synthesized['r'].isna().sum() > 0


def test_type_overrides():
    r = np.random.normal(loc=10, scale=2, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    dp = MetaExtractor.extract(df=df_original, type_overrides={'r': TypeOverride.CONTINUOUS})
    synthesizer = HighDimSynthesizer(data_panel=dp)
    assert type(synthesizer.get_values()[0]) == ContinuousValue


def test_encode():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n)})
    dp = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(data_panel=dp) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        _, df_synthesized = synthesizer.encode(df_original)
    assert df_synthesized.shape == df_original.shape


def test_encode_deterministic():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n)})
    dp = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(data_panel=dp) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized = synthesizer.encode_deterministic(df_original)
    assert df_synthesized.shape == df_original.shape
