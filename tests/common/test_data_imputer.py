import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer
from synthesized.common.data_imputer import DataImputer

NANS_PROP_TEST = 0.5


@pytest.mark.integration
def test_continuous_imputation():
    df_original = pd.DataFrame({'x': np.random.normal(loc=0, scale=1, size=1000)})
    df_original.loc[np.random.uniform(size=len(df_original)) < NANS_PROP_TEST, 'x'] = np.nan
    with HighDimSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        data_imputer = DataImputer(synthesizer=synthesizer)
        df_synthesized = data_imputer.impute_nans(df_original)

    assert df_synthesized['x'].isna().sum() == 0


@pytest.mark.integration
def test_categorical_imputation():
    df_original = pd.DataFrame({'x': np.random.choice(['a', 'b', 'c'], size=1000)})
    df_original.loc[np.random.uniform(size=len(df_original)) < NANS_PROP_TEST, 'x'] = np.nan
    with HighDimSynthesizer(df=df_original) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)
        data_imputer = DataImputer(synthesizer=synthesizer)
        df_synthesized = data_imputer.impute_nans(df_original)

    assert df_synthesized['x'].isna().sum() == 0

