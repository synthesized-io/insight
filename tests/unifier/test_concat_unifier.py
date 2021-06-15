import numpy as np
import pandas as pd
import pytest

from synthesized import MetaExtractor
from synthesized.complex.unifier.concat_unifier import ConcatUnifier

@pytest.fixture(scope="module")
def dfs():
    df1 = pd.DataFrame({'sex': ['male', 'male', 'female'],
                        'age': [10, 20, 30],
                        'income': [30000, 40000, 50000]})
    df2 = pd.DataFrame({'age': [15, 25, 30],
                        'sex': ['male', 'female', 'male'],
                        'income': [10000, 10000, 10000]})
    df3 = pd.DataFrame({'age': [15, 25, 30, 43],
                        'sex': ['female', 'female', 'male', 'male'],
                        'location': ['UK', 'US', 'US', 'UK']})
    return [df1, df2, df3]

def test_concat_dfs(dfs):
    concat_unifier = ConcatUnifier()
    concatenated_df = concat_unifier._concat_dfs(dfs=dfs)
    assert set(concatenated_df.columns) == {'sex', 'age', 'income', 'location'}
    assert set(concatenated_df['sex'].unique()) == {'male', 'female'}
    pd.testing.assert_series_equal(concatenated_df['sex'],
                                   pd.Series(['male', 'male', 'female', 'male', 'female', 'male',
                                              'female', 'female', 'male', 'male'], name='sex'))

    assert set(concatenated_df['location'].unique()) == {np.nan, 'US', 'UK'}
    pd.testing.assert_series_equal(concatenated_df['location'],
                                   pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                              'UK', 'US', 'US', 'UK'], name='location'))

@pytest.mark.slow
def test_update_and_query(dfs):
    concat_unifier = ConcatUnifier()
    df_metas = [MetaExtractor.extract(df) for df in dfs]
    concat_unifier.update(dfs, df_metas, num_iterations=10)
    new_df = pd.DataFrame({'sex': ['male', 'female'],
                           'age': [10, 20],
                           'location': ['JP', 'UK']})
    new_df_meta = MetaExtractor.extract(new_df)
    concat_unifier.update(new_df, new_df_meta, num_iterations=10)
    assert set(concat_unifier.meta['location'].categories) == {'US', 'UK', 'JP'}
    
    synth_df = concat_unifier.query(num_rows=20, columns=['age', 'location'])
    assert set(synth_df.columns)=={'age', 'location'}
    assert len(synth_df) == 20

