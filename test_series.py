import pytest

import numpy as np
import pandas as pd

from synthesized import SeriesSynthesizer


@pytest.mark.integration
def test_series_basic():
    r = np.random.normal(loc=0, scale=1, size=1000)
    df_original = pd.DataFrame({'r': r})
    with SeriesSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(series_length=100, num_series=2)
        df_synthesized2 = synthesizer.synthesize(series_lengths=[100, 50])

    assert len(df_synthesized1) == 200
    assert len(df_synthesized2) == 150

@pytest.mark.integration
def test_series_lstm1():
    r = np.random.normal(loc=0, scale=1, size=1000)
    c = np.random.choice([1, 2, 3], 1000)
    df_original = pd.DataFrame({'r': r, 's': c})
    with SeriesSynthesizer(df=df_original, lstm_mode=1, identifier_label='s') as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(series_length=100, num_series=2)
        df_synthesized2 = synthesizer.synthesize(series_lengths=[100, 50])

    assert len(df_synthesized1) == 200
    assert len(df_synthesized2) == 150


@pytest.mark.integration
def test_series_basic_lstm2():
    r = np.random.normal(loc=0, scale=1, size=1000)
    c = np.random.choice([1, 2, 3], 1000)
    df_original = pd.DataFrame({'r': r, 's': c})
    with SeriesSynthesizer(df=df_original, lstm_mode=2, identifier_label='s') as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(series_length=100, num_series=2)
        df_synthesized2 = synthesizer.synthesize(series_lengths=[100, 50])

    assert len(df_synthesized1) == 200
    assert len(df_synthesized2) == 150

