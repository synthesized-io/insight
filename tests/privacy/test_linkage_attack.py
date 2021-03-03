import logging

import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.privacy import LinkageAttack

logger = logging.getLogger(__name__)

pd.options.display.max_columns = 20


@pytest.mark.slow
def test_failed_linkage_attack():
    n = 1000

    x = np.random.normal(size=n)
    c = np.random.choice([0, 1, 2], size=n)
    y = np.array(['a', 'b', 'c'])[c]
    z = np.array([-20, 0.1, 5])[c] * x + np.random.normal(size=n)

    df_original = pd.DataFrame({'x': x, 'y': y, 'z': z})

    df_meta = MetaExtractor.extract(df_original)
    with HighDimSynthesizer(df_meta) as synthesizer:
        synthesizer.learn(num_iterations=1000, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
        assert len(df_synthesized) == len(df_original)

    la = LinkageAttack(
        df_original, t_closeness=0.3, k_distance=0.05, max_n_vulnerable=25,
        key_columns=['x', 'y'], sensitive_columns=['z']
    )
    df_attacked = la.get_attacks(df_synthesized, n_bins=5)

    assert df_attacked is None


def test_successful_linkage_attack():
    n = 1000

    x = np.random.normal(size=n)
    c = np.random.choice([0, 1, 2], size=n)
    y = np.array(['a', 'b', 'c'])[c]
    z = np.array([-10, 0.1, 5])[c] * x + np.random.normal(size=n)

    df_original = pd.DataFrame({'x': x, 'y': y, 'z': z})

    la = LinkageAttack(
        df_original, t_closeness=0.3, k_distance=0.05, max_n_vulnerable=25,
        key_columns=['x', 'y'], sensitive_columns=['z']
    )
    df_attacked = la.get_attacks(df_original, n_bins=50)
    assert df_attacked is not None
