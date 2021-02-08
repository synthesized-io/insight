from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer
from synthesized.insight.fairness import BiasMitigator, FairnessScorer
from synthesized.metadata_new import MetaExtractor
from tests.utils import progress_bar_testing


def generate_biased_data(n: int = 1000, nans_prop: Optional[float] = 0.2, date_format: str = "%Y-%m-%d"):
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
def test_bias_mitigator_mixed_types():
    sensitive_attrs = ['age', 'gender', 'DOB']
    target = 'income'
    n=1000

    df = generate_biased_data(n=n, nans_prop=0.2)

    df_meta = MetaExtractor.extract(df)
    synthesizer = HighDimSynthesizer(df_meta)
    synthesizer.learn(df_train=df, num_iterations=10)

    # Test BiasMitigator.__init__()
    n_bins = 5
    _ = BiasMitigator.from_dataframe(synthesizer, df=df, sensitive_attrs=sensitive_attrs, target=target)

    # Test bias mitigation with nans and dates
    fairness_scorer = FairnessScorer(df, sensitive_attrs=sensitive_attrs, target=target,
                                     target_n_bins=n_bins, n_bins=n_bins)
    bias_mitigator = BiasMitigator(synthesizer, fairness_scorer=fairness_scorer)
    df_unbiased = bias_mitigator.mitigate_biases_by_chunks(df, n_loops=5, progress_callback=progress_bar_testing,
                                                           produce_nans=False)

    # Number of rows
    df_unbiased = bias_mitigator.resample_df(df_unbiased, num_rows=n)
    assert len(df_unbiased) == n

    # Test bias drop
    df_unbiased_hard = bias_mitigator.drop_biases(df, progress_callback=progress_bar_testing)
    _, biases = fairness_scorer.distributions_score(df_unbiased_hard, progress_callback=progress_bar_testing)
    assert len(biases[biases["distance"].abs() > 0.05]) == 0


@pytest.mark.slow
def test_bias_mitigator_check_score():
    sensitive_attrs = ['age', 'gender']
    target = 'income'
    df = pd.read_csv("data/biased_data.csv")

    # Compute initial score
    fs = FairnessScorer(df, sensitive_attrs=sensitive_attrs, target=target)
    score_0, _ = fs.distributions_score(df, progress_callback=progress_bar_testing)

    df_meta = MetaExtractor.extract(df)
    synthesizer = HighDimSynthesizer(df_meta)
    synthesizer.learn(df_train=df, num_iterations=500)

    # Mitigate Biases
    bias_mitigator = BiasMitigator(synthesizer=synthesizer, fairness_scorer=fs)
    df_unbiased = bias_mitigator.mitigate_biases_by_chunks(df, n_loops=20, produce_nans=False,
                                                           progress_callback=progress_bar_testing)

    # Compute final score
    score_f, biases = fs.distributions_score(df_unbiased, progress_callback=progress_bar_testing)

    assert score_f > score_0

    # Number of rows
    df_unbiased = bias_mitigator.resample_df(df_unbiased, num_rows=5000)
    score_f2, biases = fs.distributions_score(df_unbiased, progress_callback=progress_bar_testing)
    assert len(df_unbiased) == 5000
    assert np.isclose(score_f, score_f2, atol=0.025)

    # Strict bias mitigation
    df_bias_drop = bias_mitigator.drop_given_biases(df_unbiased, biases=biases,
                                                    progress_callback=progress_bar_testing)

    # Compute final score
    score_f, biases_f = fs.distributions_score(df_bias_drop, progress_callback=progress_bar_testing)

    assert len(biases_f[biases_f["distance"].abs() > 0.05]) == 0
