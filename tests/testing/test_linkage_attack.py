import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.privacy import LinkageAttack


@pytest.mark.slow
def test_unittest_dataset():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n),
                                'y': np.random.choice(['a', 'b', 'c'], size=n),
                                'z': np.random.normal(size=n)})

    df_meta = MetaExtractor.extract(df_original)
    with HighDimSynthesizer(df_meta) as synthesizer:
        synthesizer.learn(num_iterations=500, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
        assert len(df_synthesized) == len(df_original)

    la = LinkageAttack(df_original, ['x', 'y'], ['z'])
    attacks = la.get_attacks(df_synthesized)
    df_attacked = la.get_attacked_rows(attacks)
    assert len(df_attacked) == 0
