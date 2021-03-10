import numpy as np
import pandas as pd
import pytest

from synthesized.complex import HighDimSynthesizer
from synthesized.metadata_new.factory import MetaExtractor
from synthesized.testing import UtilityTesting


@pytest.fixture
def simple_df():
    np.random.seed(6235901)
    df = pd.DataFrame({
        'string': np.random.choice(['A','B','C','D','E'], size=1000),
        'bool': np.random.choice([False, True], size=1000).astype('?'),
        'date': pd.to_datetime(18_000 + np.random.normal(500, 50, size=1000).astype(int), unit='D'),
        'int': [n for n in [0, 1, 2, 3, 4, 5] for i in range([50, 50, 0, 200, 400, 300][n])],
        'float': np.random.normal(0.0, 1.0, size=1000),
        'int_bool': np.random.choice([0, 1], size=1000),
        'date_sparse': pd.to_datetime(18_000 + 5 * np.random.normal(500, 50, size=1000).astype(int), unit='D')
    })
    return df


def test_utility(simple_df):
    df_meta = MetaExtractor.extract(simple_df)
    synth = HighDimSynthesizer(df_meta)
    testing = UtilityTesting(synth, simple_df, simple_df, simple_df)
    assert set(testing.continuous) == {'float'}
    assert set(testing.categorical) == {'string', 'bool', 'int', 'int_bool'}
