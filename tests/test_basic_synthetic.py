import pytest

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from synthesized import BasicSynthesizer


@pytest.mark.integration
def test_continuous_variable_generation():
    r = np.random.normal(loc=5000, scale=1000, size=1000)
    df_original = pd.DataFrame({'r': r})
    with BasicSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.integration
def test_categorical_similarity_variable_generation():
    r = np.random.normal(loc=10, scale=2, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    with BasicSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.integration
def test_categorical_variable_generation():
    r = np.random.normal(loc=5, scale=1, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    with BasicSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=10000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    assert distribution_distance < 0.3
