import numpy as np
import pandas as pd
import pytest

from synthesized import SeriesSynthesizer, MetaExtractor
from synthesized.config import SeriesConfig


@pytest.mark.skip("Not updated with new metas.")
@pytest.mark.slow
def test_series_basic():
    r = np.random.normal(loc=0, scale=1, size=1000)
    df_original = pd.DataFrame({'r': r})
    df_meta = MetaExtractor.extract(df=df_original)
    with SeriesSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(num_rows=100, num_series=2)

    assert len(df_synthesized1) == 200


@pytest.mark.skip("Not updated with new metas.")
@pytest.mark.slow
def test_series_synthesis_identifier():
    r = np.random.normal(loc=0, scale=1, size=900)
    t = pd.date_range(start='01-01-2020', periods=300).repeat(3)
    c = np.array([1, 2, 3]*300)
    df_original = pd.DataFrame({'r': r, 'c': c, 't': t})
    df_meta = MetaExtractor.extract(df=df_original, id_index='c', time_index='t')
    with SeriesSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(num_rows=100, num_series=3)

    assert len(df_synthesized1) == 300
    assert df_synthesized1['c'].nunique() == 3


@pytest.mark.skip("Not updated with new metas.")
@pytest.mark.slow
def test_series_lstm():
    r = np.random.normal(loc=0, scale=1, size=900)
    t = pd.date_range(start='01-01-2020', periods=300).repeat(3)
    c = np.array([1, 2, 3]*300)
    df_original = pd.DataFrame({'r': r, 'c': c, 't': t})
    config = SeriesConfig(lstm_mode='lstm')
    df_meta = MetaExtractor.extract(df=df_original, id_index='c', time_index='t')
    with SeriesSynthesizer(df_meta=df_meta, config=config) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(num_rows=100, num_series=2)

    assert len(df_synthesized1) == 200


@pytest.mark.skip("Not updated with new metas.")
@pytest.mark.slow
def test_series_basic_vrae():
    r = np.random.normal(loc=0, scale=1, size=900)
    t = pd.date_range(start='01-01-2020', periods=300).repeat(3)
    c = np.array([1, 2, 3] * 300)
    df_original = pd.DataFrame({'r': r, 'c': c, 't': t})
    config = SeriesConfig(lstm_mode='vrae')
    df_meta = MetaExtractor.extract(df=df_original, id_index='c', time_index='t')
    with SeriesSynthesizer(df_meta=df_meta, config=config) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(num_rows=100, num_series=2)

    assert len(df_synthesized1) == 200


@pytest.mark.skip("Not updated with new metas.")
@pytest.mark.slow
def test_series_rdssm():
    r = np.random.normal(loc=0, scale=1, size=900)
    t = pd.date_range(start='01-01-2020', periods=300).repeat(3)
    c = np.array([1, 2, 3] * 300)
    df_original = pd.DataFrame({'r': r, 'c': c, 't': t})
    config = SeriesConfig(lstm_mode='rdssm')
    df_meta = MetaExtractor.extract(df=df_original, id_index='c', time_index='t')
    with SeriesSynthesizer(df_meta=df_meta, config=config) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        df_synthesized1 = synthesizer.synthesize(num_rows=100, num_series=2)

    assert len(df_synthesized1) == 200


# TODO: Fails in CircleCI, works in local
@pytest.mark.skip(reason="Fails in CircleCI, works in local.")
@pytest.mark.slow
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
@pytest.mark.slow
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
@pytest.mark.slow
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
