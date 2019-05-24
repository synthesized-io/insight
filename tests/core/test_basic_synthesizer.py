import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from synthesized.core import BasicSynthesizer


@pytest.mark.integration
def test_continuous_variable_generation():
    r = np.random.normal(loc=5000, scale=1000, size=1000)
    data = pd.DataFrame({'r': r})
    with BasicSynthesizer(data=data) as synthesizer:
        synthesizer.learn(num_iterations=1000, data=data)
        synthesized = synthesizer.synthesize(n=len(data))
    distribution_distance = ks_2samp(data['r'], synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.integration
def test_categorical_similarity_variable_generation():
    r = np.random.normal(loc=10, scale=2, size=1000)
    data = pd.DataFrame({'r': list(map(int, r))})
    with BasicSynthesizer(data=data) as synthesizer:
        synthesizer.learn(num_iterations=1000, data=data)
        synthesized = synthesizer.synthesize(n=len(data))
    distribution_distance = ks_2samp(data['r'], synthesized['r'])[0]
    assert distribution_distance < 0.3


@pytest.mark.integration
def test_categorical_variable_generation():
    r = np.random.normal(loc=5, scale=1, size=1000)
    data = pd.DataFrame({'r': list(map(int, r))})
    with BasicSynthesizer(data=data) as synthesizer:
        synthesizer.learn(num_iterations=1000, data=data)
        synthesized = synthesizer.synthesize(n=len(data))
    distribution_distance = ks_2samp(data['r'], synthesized['r'])[0]
    assert distribution_distance < 0.3
