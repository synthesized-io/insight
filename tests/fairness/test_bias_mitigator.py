from typing import Optional

import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.insight.fairness import BiasMitigator, FairnessScorer
from synthesized.testing.utils import testing_progress_bar


def generate_biased_data(n: int = 5000, nans_prop: Optional[float] = 0.05):
    x = np.random.randn(n)
    y = x + 0.5 * np.random.randn(n)
    x_nans = np.where(np.random.rand(n) < nans_prop, np.nan, np.random.randn(n)) if nans_prop else np.random.randn(n)

    df = pd.DataFrame({
        'x_nans': x_nans,
        'age': [int(abs(xx * 15 + 50)) for xx in x],
        'gender': ['M' if (yy + np.random.randn() * 0.5) > 0.5 else 'F' for yy in y],
        'income': [int(abs(yy * 5_000 + 40_000)) for yy in y]
    })
    return df


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_bins",
    [
        pytest.param(2, id="binary_target"),
        pytest.param(5, id="multinomial_target"),
    ]
)
def test_bias_mitigator(n_bins):
    sensitive_attrs = ['age', 'gender']
    target = 'income'
    df = generate_biased_data()

    fs_0 = FairnessScorer(df, sensitive_attrs=sensitive_attrs, target=target)
    score_0, df_biases_0 = fs_0.distributions_score(progress_callback=testing_progress_bar)

    df_meta = MetaExtractor.extract(df)
    synthesizer = HighDimSynthesizer(df_meta)
    synthesizer.learn(df_train=df, num_iterations=500)

    # Test __init__()
    _ = BiasMitigator(synthesizer, fairness_scorer=FairnessScorer(df, sensitive_attrs=sensitive_attrs, target=target,
                                                                  target_n_bins=n_bins, n_bins=n_bins))
    # Test from_dataframe()
    bias_mitigator = BiasMitigator.from_dataframe(synthesizer, df=df, sensitive_attrs=sensitive_attrs, target=target,
                                                  n_bins=n_bins)
    df_unbiased = bias_mitigator.mitigate_biases_by_chunks(df, n_loops=50, progress_callback=testing_progress_bar,
                                                           produce_nans=False)

    fs_f = FairnessScorer(df_unbiased, sensitive_attrs=sensitive_attrs, target=target)
    score_f, df_biases_f = fs_f.distributions_score(progress_callback=testing_progress_bar)

    assert score_f > score_0
    assert not df_unbiased.isna().any(axis=None)