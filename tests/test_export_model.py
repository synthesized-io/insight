import shutil
import tempfile
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.common.values import ContinuousValue
from synthesized.config import HighDimConfig
from synthesized.metadata import TypeOverride

atol = 0.05


def export_model_given_df(df_original: pd.DataFrame, num_iterations: int = 500, highdim_kwargs: Dict[str, Any] = None,
                          synthesize_kwargs: Dict[str, Any] = None):
    highdim_kwargs = dict() if highdim_kwargs is None else highdim_kwargs
    synthesize_kwargs = dict() if synthesize_kwargs is None else synthesize_kwargs

    temp_dir = tempfile.mkdtemp()
    temp_fname = temp_dir + 'synthesizer.txt'

    df_meta = MetaExtractor.extract(df=df_original)
    config = HighDimConfig(**highdim_kwargs)
    with HighDimSynthesizer(df_meta=df_meta, config=config) as synthesizer:
        synthesizer.learn(num_iterations=num_iterations, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original), **synthesize_kwargs)

        with open(temp_fname, 'wb') as f:
            synthesizer.export_model(f)

    with open(temp_fname, 'rb') as f:
        synthesizer2 = HighDimSynthesizer.import_model(f)

    df_synthesized2 = synthesizer2.synthesize(num_rows=len(df_original), **synthesize_kwargs)
    shutil.rmtree(temp_dir)

    return df_synthesized, df_synthesized2


@pytest.mark.slow
def test_continuous_variable_generation():
    r = np.random.normal(loc=5000, scale=1000, size=1000)
    df_original = pd.DataFrame({'r': r})
    df_synthesized, df_synthesized2 = export_model_given_df(df_original)

    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    distribution_distance2 = ks_2samp(df_original['r'], df_synthesized2['r'])[0]
    assert distribution_distance2 < 0.3
    assert np.isclose(distribution_distance, distribution_distance2, atol=atol)


@pytest.mark.slow
def test_categorical_similarity_variable_generation():
    r = np.random.normal(loc=10, scale=2, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    df_synthesized, df_synthesized2 = export_model_given_df(df_original)

    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    distribution_distance2 = ks_2samp(df_original['r'], df_synthesized2['r'])[0]
    assert distribution_distance2 < 0.3
    assert np.isclose(distribution_distance, distribution_distance2, atol=atol)


@pytest.mark.slow
def test_categorical_variable_generation():
    r = np.random.normal(loc=5, scale=1, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    df_synthesized, df_synthesized2 = export_model_given_df(df_original)

    distribution_distance = ks_2samp(df_original['r'], df_synthesized['r'])[0]
    distribution_distance2 = ks_2samp(df_original['r'], df_synthesized2['r'])[0]
    assert distribution_distance2 < 0.3
    assert np.isclose(distribution_distance, distribution_distance2, atol=atol)


@pytest.mark.slow
def test_nan_producing():
    r = np.random.normal(loc=0, scale=1, size=1000)
    indices = np.random.choice(np.arange(r.size), replace=False, size=int(r.size * 0.2))
    r[indices] = np.nan
    df_original = pd.DataFrame({'r': r})

    df_synthesized, df_synthesized2 = export_model_given_df(df_original, synthesize_kwargs=dict(produce_nans=True))

    assert df_synthesized['r'].isna().sum() > 0
    assert np.isclose(np.sum(np.isnan(df_synthesized['r'])) / len(df_synthesized),
                      np.sum(np.isnan(df_synthesized2['r'])) / len(df_synthesized2),
                      atol=atol)


@pytest.mark.slow
def test_type_overrides():
    r = np.random.normal(loc=10, scale=2, size=1000)
    df_original = pd.DataFrame({'r': list(map(int, r))})
    df_meta = MetaExtractor.extract(df=df_original, type_overrides={'r': TypeOverride.CONTINUOUS})
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)

    temp_dir = tempfile.mkdtemp()
    temp_fname = temp_dir + 'synthesizer.txt'

    with open(temp_fname, 'wb') as f:
        synthesizer.export_model(f)

    with open(temp_fname, 'rb') as f:
        synthesizer2 = HighDimSynthesizer.import_model(f)
    shutil.rmtree(temp_dir)

    assert type(synthesizer2.get_values()[0]) == ContinuousValue
