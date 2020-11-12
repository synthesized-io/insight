from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.insight.fairness import BiasMitigator, FairnessScorer
from synthesized.testing.utils import testing_progress_bar


def generate_biased_data(n: int = 500, nans_prop: Optional[float] = 0.25, date_format: str = "%Y-%m-%d"):
    x = np.random.randn(n)
    y = x + 0.5 * np.random.randn(n)

    df = pd.DataFrame({
        'x': np.random.randn(n),
        'age': [int(abs(xx * 15 + 50)) for xx in x],
        'gender': ['M' if (yy + np.random.randn() * 0.5) > 0.5 else 'F' for yy in y],
        'income': [int(abs(yy * 5_000 + 40_000)) for yy in y],
        'DOB': [datetime.strftime(d, date_format) for d in pd.to_datetime(np.random.randn(n) * 1e18)]
    })
    if nans_prop is not None and nans_prop > 0:
        for c in df.columns:
            df[c] = np.where(np.random.rand(n) < nans_prop, np.nan, df[c])
    return df


@pytest.mark.slow
def test_bias_mitigator_multinomial_target():
    sensitive_attrs = ['age', 'gender', 'DOB']
    target = 'income'
    df = generate_biased_data()

    fs_0 = FairnessScorer(df, sensitive_attrs=sensitive_attrs, target=target)
    score_0, df_biases_0 = fs_0.distributions_score(progress_callback=testing_progress_bar)

    df_meta = MetaExtractor.extract(df)
    synthesizer = HighDimSynthesizer(df_meta)
    synthesizer.learn(df_train=df, num_iterations=500)

    # Test __init__()
    n_bins = 5
    _ = BiasMitigator(synthesizer, fairness_scorer=FairnessScorer(df, sensitive_attrs=sensitive_attrs, target=target,
                                                                  target_n_bins=n_bins, n_bins=n_bins))
    # Test from_dataframe()
    bias_mitigator = BiasMitigator.from_dataframe(synthesizer, df=df, sensitive_attrs=sensitive_attrs, target=target)
    df_unbiased = bias_mitigator.mitigate_biases_by_chunks(df, n_loops=50, progress_callback=testing_progress_bar,
                                                           produce_nans=False)

    fs_f = FairnessScorer(df_unbiased, sensitive_attrs=sensitive_attrs, target=target)
    score_f, df_biases_f = fs_f.distributions_score(progress_callback=testing_progress_bar)

    assert score_f > score_0
    assert not df_unbiased.isna().any(axis=None)
