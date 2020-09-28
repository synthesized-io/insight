import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.complex import ConditionalSampler
from synthesized.testing.utils import testing_progress_bar


@pytest.mark.slow
def test_categorical_continuous_sampling():
    num_rows = 1000
    df_original = pd.DataFrame({'x': np.random.randn(num_rows), 'y': np.random.choice(['a', 'b', 'c'], num_rows)})

    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)

    # Single marginal
    marginals = {'y': {'a': 0.6, 'b': 0.3, 'c': 0.1}}
    conditional_sampler = ConditionalSampler(synthesizer)
    df_synthesized = conditional_sampler.synthesize(num_rows=num_rows, explicit_marginals=marginals,
                                                    progress_callback=testing_progress_bar)

    value_counts_y = df_synthesized['y'].value_counts(normalize=True)
    for k, marginal_k in marginals['y'].items():
        assert np.isclose(marginal_k, value_counts_y[k], atol=0.02)

    # Multiple marginals
    marginals = {'y': {'a': 0.6, 'b': 0.3, 'c': 0.1},
                 'x': {'[-10.0, 0.0)': 0.9, '[0.0, 10.0)': 0.1}}
    conditional_sampler = ConditionalSampler(synthesizer)
    df_synthesized = conditional_sampler.synthesize(num_rows=num_rows, explicit_marginals=marginals,
                                                    progress_callback=testing_progress_bar)

    value_counts_y = df_synthesized['y'].value_counts(normalize=True)
    for k, marginal_k in marginals['y'].items():
        assert np.isclose(marginal_k, value_counts_y[k], atol=0.02)

    assert np.isclose(len(df_synthesized[df_synthesized['x'] < 0]) / num_rows, 0.9, atol=0.02)
    assert np.isclose(len(df_synthesized[df_synthesized['x'] >= 0]) / num_rows, 0.1, atol=0.02)


@pytest.mark.slow
def test_alter_distributions():
    num_rows = 1000
    df_original = pd.DataFrame({'x': np.random.randn(num_rows), 'y': np.random.choice(['a', 'b', 'c'], num_rows)})

    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)

    # Single marginal
    marginals = {'y': {'a': 0.6, 'b': 0.3, 'c': 0.1}}
    conditional_sampler = ConditionalSampler(synthesizer)
    df_synthesized = conditional_sampler.alter_distributions(df_original, num_rows=num_rows,
                                                             explicit_marginals=marginals,
                                                             progress_callback=testing_progress_bar)

    value_counts_y = df_synthesized['y'].value_counts(normalize=True)
    for k, marginal_k in marginals['y'].items():
        assert np.isclose(marginal_k, value_counts_y[k], atol=0.02)

    # Multiple marginals
    marginals = {'y': {'a': 0.6, 'b': 0.3, 'c': 0.1},
                 'x': {'[-10.0, 0.0)': 0.9, '[0.0, 10.0)': 0.1}}
    conditional_sampler = ConditionalSampler(synthesizer)
    df_synthesized = conditional_sampler.alter_distributions(df_original, num_rows=num_rows,
                                                             explicit_marginals=marginals,
                                                             progress_callback=testing_progress_bar)

    value_counts_y = df_synthesized['y'].value_counts(normalize=True)
    for k, marginal_k in marginals['y'].items():
        assert np.isclose(marginal_k, value_counts_y[k], atol=0.02)

    assert np.isclose(len(df_synthesized[df_synthesized['x'] < 0]) / num_rows, 0.9, atol=0.02)
    assert np.isclose(len(df_synthesized[df_synthesized['x'] >= 0]) / num_rows, 0.1, atol=0.02)


@pytest.mark.slow
def test_alter_distributions_nans():
    num_rows = 1000
    x = np.random.normal(loc=0, scale=1, size=num_rows)
    indices = np.random.choice(np.arange(x.size), replace=False, size=int(x.size * 0.2))
    x[indices] = np.nan
    df_original = pd.DataFrame({'x': x,
                                'y': np.random.choice(['a', 'b', 'c'], num_rows)})

    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=100, df_train=df_original)

    # Single marginal
    marginals = {'y': {'a': 0.6, 'b': 0.3, 'c': 0.1}}
    conditional_sampler = ConditionalSampler(synthesizer)

    # WITH nans
    df_synthesized_w_nans = conditional_sampler.alter_distributions(
        df_original, num_rows=num_rows, produce_nans=True, explicit_marginals=marginals,
        progress_callback=testing_progress_bar)

    assert df_synthesized_w_nans.isna().values.any()
    value_counts_y = df_synthesized_w_nans['y'].value_counts(normalize=True)
    for k, marginal_k in marginals['y'].items():
        assert np.isclose(marginal_k, value_counts_y[k], atol=0.02)

    # WITHOUT nans
    df_synthesized_wo_nans = conditional_sampler.alter_distributions(
        df_original, num_rows=num_rows, produce_nans=False, explicit_marginals=marginals,
        progress_callback=testing_progress_bar)

    assert not df_synthesized_wo_nans.isna().values.any()
    value_counts_y = df_synthesized_wo_nans['y'].value_counts(normalize=True)
    for k, marginal_k in marginals['y'].items():
        assert np.isclose(marginal_k, value_counts_y[k], atol=0.02)
