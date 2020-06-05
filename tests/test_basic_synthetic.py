import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from synthesized import HighDimSynthesizer
from synthesized.values import TypeOverride, ContinuousValue


@pytest.mark.integration
def test_continuous_variable_generation():
    r = np.random.normal(loc=5000, scale=1000, size=1000)
    df_original = pd.DataFrame({'r': r})
    with HighDimSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.integration
def test_categorical_similarity_variable_generation():
    r = np.random.normal(loc=10, scale=2, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    with HighDimSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.integration
def test_categorical_variable_generation():
    r = np.random.normal(loc=5, scale=1, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    with HighDimSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.integration
def test_nan_producing():
    r = np.random.normal(loc=5000, scale=1000, size=1000)
    indices = np.random.choice(np.arange(r.size), replace=False, size=int(r.size * 0.2))
    r[indices] = np.nan
    df_original = pd.DataFrame({'r': r})
    with HighDimSynthesizer(df=df_original, produce_nans_for={'r'}) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    assert np.isnan(df_synthesized['r']).any()


def test_type_overrides():
    r = np.random.normal(loc=10, scale=2, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    synthesizer = HighDimSynthesizer(df=df_original, type_overrides={'r': TypeOverride.CONTINUOUS})
    assert type(synthesizer.get_values()[0]) == ContinuousValue
