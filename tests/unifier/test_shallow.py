import numpy as np
import pandas as pd

from synthesized import MetaExtractor
from synthesized.complex.unifier import ShallowOracle


def test_empty():

    ora = ShallowOracle()
    df_s: pd.DataFrame = ora.query(columns=['A', 'B', 'C'], num_rows=1000)
    assert df_s.dropna().empty


def test_same_columns_new_range():

    df_1 = pd.DataFrame({
        'A': np.random.choice(['a', 'b', 'c'], size=1000),
        'B': np.random.normal(-3.0, scale=1.0, size=1000)
    })
    df_2 = pd.DataFrame({
        'A': np.random.choice(['c', 'd', 'e'], size=4000),
        'B': np.random.normal(3.0, scale=1.0, size=4000)
    })

    ora = ShallowOracle()
    df_meta_1 = MetaExtractor.extract(df_1)
    ora.update(dfs=df_1, df_metas=df_meta_1)
    df_s1 = ora.query(columns=['A', 'B'], num_rows=1000)

    df_meta_2 = MetaExtractor.extract(df_2)
    ora.update(dfs=[df_2], df_metas=[df_meta_2])
    df_s2 = ora.query(columns=['A', 'B'], num_rows=3000)

    assert df_s2['A'].nunique() == 5
    assert df_s2['B'].mean() > 0
