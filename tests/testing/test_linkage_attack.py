import pandas as pd
import numpy as np

from synthesized import HighDimSynthesizer
from synthesized.privacy import LinkageAttack


def test_unittest_dataset():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n),
                                'y': np.random.choice(['a', 'b', 'c'], size=n),
                                'z': np.random.normal(size=n)})

    with HighDimSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=500, df_train=df_original)
        df_synthesized = synthesizer.synthesize(num_rows=len(df_original))
        assert len(df_synthesized) == len(df_original)

    la = LinkageAttack(df_original, ['x', 'y'], ['z'])
    attacks = la.get_attacks(df_synthesized)
    df_attacked = la.get_attacked_rows(attacks)
    assert len(df_attacked) == 0
