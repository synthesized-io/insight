import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer, MetaExtractor
from synthesized.complex import DataImputer

NANS_PROP_TEST = 0.5


def test_continuous_nans_imputation():
    df_original = pd.DataFrame({'x': np.random.normal(loc=0, scale=1, size=1000)})
    df_original.loc[np.random.uniform(size=len(df_original)) < NANS_PROP_TEST, 'x'] = np.nan
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)

    data_imputer = DataImputer(synthesizer=synthesizer)
    df_synthesized = data_imputer.impute_nans(df_original)
    assert df_synthesized['x'].isna().sum() == 0

    data_imputer.impute_nans(df_original, inplace=True)
    assert df_original['x'].isna().sum() == 0


def test_categorical_nans_imputation():
    df_original = pd.DataFrame({'x': np.random.choice(['a', 'b', 'c'], size=1000)})
    df_original.loc[np.random.uniform(size=len(df_original)) < NANS_PROP_TEST, 'x'] = np.nan
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=10, df_train=df_original)

    data_imputer = DataImputer(synthesizer=synthesizer)
    df_synthesized = data_imputer.impute_nans(df_original)
    assert df_synthesized['x'].isna().sum() == 0

    data_imputer.impute_nans(df_original, inplace=True)
    assert df_original['x'].isna().sum() == 0


def test_continuous_outliers_imputation():
    n = 1000
    x = np.random.normal(loc=0, scale=1, size=n)
    df_original = pd.DataFrame({
        'x': np.where(np.random.uniform(size=n) < 0.98, x, np.random.normal(loc=0, scale=50, size=n)),
        'y': 0.5 * x
    })
    df_meta = MetaExtractor.extract(df=df_original)
    with HighDimSynthesizer(df_meta=df_meta) as synthesizer:
        synthesizer.learn(num_iterations=1000, df_train=df_original)

    data_imputer = DataImputer(synthesizer=synthesizer)
    df_synthesized = data_imputer.impute_outliers(df_original, outliers_percentile=0.05)
    assert np.sum(df_synthesized['x'].values > 10) == 0

    data_imputer.impute_outliers(df_original, inplace=True, outliers_percentile=0.05)
    assert np.sum(df_original['x'].values > 10) == 0


def test_mixed_dtypes_nan_imputation():
    num_iterations = 50

    df_original = pd.read_csv("data/credit.csv")
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.__enter__()
    synthesizer.learn(num_iterations=num_iterations, df_train=df_original)
    df_synthesized = synthesizer.synthesize(num_rows=10)
    print(df_original.shape, df_synthesized.shape)  # (9999, 11) (10, 11)
    print([df_original[col].dtype for col in df_original.columns])

    data_imputer = DataImputer(synthesizer)
    data_imputer.impute_nans(df_original)  # TypeError: Cannot do inplace boolean setting on mixed-types with a non np.nan value

    df = pd.concat([df_original, df_synthesized])
    print(df.shape)  # (10009, 11)
    data_imputer.impute_nans(df)  # RecursionError: maximum recursion depth exceeded