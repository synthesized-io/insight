import numpy as np
import pandas as pd

from synthesized import HighDimSynthesizer
from synthesized.common import ConditionalSampler


def test_categorical_sampling():
    num_rows = 1000
    df_original = pd.DataFrame({'x': np.random.randn(num_rows), 'y': np.random.choice(['a', 'b', 'c'], num_rows)})
    with HighDimSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=1000, df_train=df_original)
        y_marginals = {'a': 0.6, 'b': 0.3, 'c': 0.1}
        conditional_sampler = ConditionalSampler(synthesizer, ('y', y_marginals))
        df_synthesized = conditional_sampler.synthesize(num_rows=num_rows)

    value_counts = df_synthesized['y'].value_counts() / num_rows

    for k in y_marginals.keys():
        assert np.isclose(y_marginals[k], value_counts[k], atol=0.02)


def test_continuous_sampling():
    num_rows = 1000
    df_original = pd.DataFrame({'x': np.random.randn(num_rows), 'y': np.random.choice(['a', 'b', 'c'], num_rows)})
    with HighDimSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=1000, df_train=df_original)
        x_marginals = {'[-10.0, 0.0)': 0.9, '[0.0, 10.0)': 0.1}
        conditional_sampler = ConditionalSampler(synthesizer, ('x', x_marginals))
        df_synthesized = conditional_sampler.synthesize(num_rows=num_rows)

    assert np.isclose(len(df_synthesized[df_synthesized['x'] < 0]) / num_rows, 0.9, atol=0.02)
    assert np.isclose(len(df_synthesized[df_synthesized['x'] >= 0]) / num_rows, 0.1, atol=0.02)

