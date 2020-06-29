import numpy as np
import pandas as pd
import pytest

from synthesized import SeriesSynthesizer, MetaExtractor
from synthesized.complex.series import SeriesConfig


@pytest.mark.integration
def test_series_basic():
    r = np.random.normal(loc=0, scale=1, size=1000)
    df_original = pd.DataFrame({'r': r})
    df_meta = MetaExtractor.extract(df=df_original)
    with SeriesSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(series_length=100, num_series=2)
        df_synthesized2 = synthesizer.synthesize(series_lengths=[100, 50])

    assert len(df_synthesized1) == 200
    assert len(df_synthesized2) == 150


@pytest.mark.integration
def test_series_synthesis_identifier():
    r = np.random.normal(loc=0, scale=1, size=1000)
    c = np.random.choice([1, 2, 3], 1000)
    df_original = pd.DataFrame({'r': r, 's': c})
    df_meta = MetaExtractor.extract(df=df_original, id_index='s')
    with SeriesSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(series_length=100, num_series=3)
        df_synthesized2 = synthesizer.synthesize(series_lengths=[100, 50, 25])

    assert len(df_synthesized1) == 300
    assert df_synthesized1['s'].nunique() == 3
    assert len(df_synthesized2) == 175
    assert df_synthesized2['s'].nunique() == 3


@pytest.mark.integration
def test_series_lstm():
    r = np.random.normal(loc=0, scale=1, size=1000)
    c = np.random.choice([1, 2, 3], 1000)
    df_original = pd.DataFrame({'r': r, 's': c})
    config = SeriesConfig(lstm_mode='lstm')
    df_meta = MetaExtractor.extract(df=df_original, id_index='s')
    with SeriesSynthesizer(df_meta=df_meta, config=config) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(series_length=100, num_series=2)
        df_synthesized2 = synthesizer.synthesize(series_lengths=[100, 50])

    assert len(df_synthesized1) == 200
    assert len(df_synthesized2) == 150


@pytest.mark.integration
def test_series_basic_vrae():
    r = np.random.normal(loc=0, scale=1, size=1000)
    c = np.random.choice([1, 2, 3], 1000)
    df_original = pd.DataFrame({'r': r, 's': c})
    config = SeriesConfig(lstm_mode='vrae')
    df_meta = MetaExtractor.extract(df=df_original, id_index='s')
    with SeriesSynthesizer(df_meta=df_meta, config=config) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(series_length=100, num_series=2)
        df_synthesized2 = synthesizer.synthesize(series_lengths=[100, 50])

    assert len(df_synthesized1) == 200
    assert len(df_synthesized2) == 150


@pytest.mark.integration
def test_series_rdssm():
    r = np.random.normal(loc=0, scale=1, size=1000)
    c = np.random.choice([1, 2, 3], 1000)
    df_original = pd.DataFrame({'r': r, 's': c})
    config = SeriesConfig(lstm_mode='rdssm')
    df_meta = MetaExtractor.extract(df=df_original, id_index='s')
    with SeriesSynthesizer(df_meta=df_meta, config=config) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(series_length=100, num_series=2)
        df_synthesized2 = synthesizer.synthesize(series_lengths=[100, 50])

    assert len(df_synthesized1) == 200
    assert len(df_synthesized2) == 150


# TODO: Fails in CircleCI, works in local
@pytest.mark.skip(reason="Fails in CircleCI, works in local.")
@pytest.mark.integration
def test_series_encode_lstm():
    r = np.random.normal(loc=0, scale=1, size=100)
    c = np.random.choice([1, 2, 3], 100)
    df_original = pd.DataFrame({'r': r, 's': c})
    config = SeriesConfig(lstm_mode='lstm')
    df_meta = MetaExtractor.extract(df=df_original)
    with SeriesSynthesizer(df_meta=df_meta, config=config) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_encoded, df_synthesized = synthesizer.encode(df_original, n_forecast=50)

    assert len(df_synthesized) == 150


# TODO: Fails in CircleCI, works in local
@pytest.mark.skip(reason="Fails in CircleCI, works in local.")
@pytest.mark.integration
def test_series_encode_vrae():
    r = np.random.normal(loc=0, scale=1, size=100)
    c = np.random.choice([1, 2, 3], 100)
    df_original = pd.DataFrame({'r': r, 's': c})
    config = SeriesConfig(lstm_mode='vrae')
    df_meta = MetaExtractor.extract(df=df_original)
    with SeriesSynthesizer(df_meta=df_meta, config=config) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_encoded, df_synthesized = synthesizer.encode(df_original, n_forecast=50)

    assert len(df_synthesized) == 150


@pytest.mark.skip(reason="Encoding not implemented for DSS.")
@pytest.mark.integration
def test_series_encode_dss():
    r = np.random.normal(loc=0, scale=1, size=100)
    c = np.random.choice([1, 2, 3], 100)
    df_original = pd.DataFrame({'r': r, 's': c})
    config = SeriesConfig(lstm_mode='rdssm')
    df_meta = MetaExtractor.extract(df=df_original)
    with SeriesSynthesizer(df_meta=df_meta, config=config) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_encoded, df_synthesized = synthesizer.encode(df_original, n_forecast=50)

    assert len(df_synthesized) == 150
