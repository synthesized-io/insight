import numpy as np
import pandas as pd
import pytest

from synthesized import HighDimSynthesizer
from synthesized.complex import DataImputer
from synthesized.metadata.factory import MetaExtractor
from tests.utils import progress_bar_testing

NANS_PROP_TEST = 0.5


@pytest.mark.slow
def test_continuous_nans_imputation():
    df_original = pd.DataFrame({'x': np.random.normal(loc=0, scale=1, size=1000)})
    df_original.loc[np.random.uniform(size=len(df_original)) < NANS_PROP_TEST, 'x'] = np.nan
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)

    data_imputer = DataImputer(synthesizer=synthesizer)
    df_synthesized = data_imputer.impute_nans(df_original)
    assert df_synthesized['x'].isna().sum() == 0

    data_imputer.impute_nans(df_original, inplace=True, progress_callback=progress_bar_testing)
    assert df_original['x'].isna().sum() == 0


@pytest.mark.slow
def test_categorical_nans_imputation():
    df_original = pd.DataFrame({'x': np.random.choice(['a', 'b', 'c'], size=1000)})
    df_original.loc[np.random.uniform(size=len(df_original)) < NANS_PROP_TEST, 'x'] = np.nan
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)

    data_imputer = DataImputer(synthesizer=synthesizer)
    df_synthesized = data_imputer.impute_nans(df_original)
    assert df_synthesized['x'].isna().sum() == 0

    # Make sure everything not imputed is same as in original
    assert all((df_original == df_synthesized) == ~df_original.isna())

    data_imputer.impute_nans(df_original, inplace=True, progress_callback=progress_bar_testing)
    assert df_original['x'].isna().sum() == 0


@pytest.mark.slow
def test_mixed_dtypes_nan_imputation():
    num_iterations = 50

    df_original = pd.read_csv("data/credit.csv")
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=num_iterations, df_train=df_original)
    df_synthesized = synthesizer.synthesize(num_rows=10)
    print(df_original.shape, df_synthesized.shape)  # (9999, 11) (10, 11)
    print([df_original[col].dtype for col in df_original.columns])

    data_imputer = DataImputer(synthesizer)
    data_imputer.impute_nans(df_original, progress_callback=progress_bar_testing)  # TypeError: Cannot do inplace boolean setting on mixed-types with a non np.nan value

    df = pd.concat([df_original, df_synthesized])
    print(df.shape)  # (10009, 11)
    data_imputer.impute_nans(df, progress_callback=progress_bar_testing)  # RecursionError: maximum recursion depth exceeded


@pytest.mark.slow
def test_mask_imputation():
    n = 1000
    df_original = pd.DataFrame({
        'x': np.random.normal(loc=0, scale=1, size=n),
        'y': np.random.choice(['a', 'b', 'c'], size=n),
    })

    mask = pd.DataFrame({
        'x': np.random.choice([True, False], size=n, p=[0.6, 0.4]),
        'y': np.random.choice([True, False], size=n, p=[0.2, 0.8]),
    })

    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)

    data_imputer = DataImputer(synthesizer=synthesizer)

    df_imputed = data_imputer.impute_mask(df_original, mask=mask)
    assert (df_original == df_imputed)[~mask].sum().sum() == mask.count().sum() - mask.sum().sum()

    df_original_copy = df_original.copy()
    data_imputer.impute_mask(df_original_copy, mask=mask, progress_callback=progress_bar_testing, inplace=True)
    assert (df_original == df_original_copy)[~mask].sum().sum() == mask.count().sum() - mask.sum().sum()

    empty_mask = pd.DataFrame({'x': np.zeros(n), 'y': np.zeros(n)}, dtype=bool)
    _ = data_imputer.impute_mask(df_original_copy, mask=empty_mask)

    with pytest.raises(ValueError):
        wrong_mask = pd.DataFrame({'x': np.random.choice([True, False], size=10, p=[0.6, 0.4])})
        _ = data_imputer.impute_mask(df_original_copy, mask=wrong_mask)


@pytest.mark.slow
def test_outliers_imputation():
    n = 1000
    df_original = pd.DataFrame({
        'x': np.where(
            np.random.uniform(size=n) > 0.1,
            np.random.normal(loc=0, scale=1, size=n),
            np.random.normal(loc=0, scale=1000, size=n)
        )
    })
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)
    synthesizer.learn(num_iterations=10, df_train=df_original)

    data_imputer = DataImputer(synthesizer=synthesizer)
    df_out = data_imputer.impute_outliers(df_original)
    data_imputer.impute_outliers(df_original, inplace=True, progress_callback=progress_bar_testing)


def test_repr():
    n = 1000
    df_original = pd.DataFrame({'x': np.random.normal(size=n), 'y': np.random.choice(['a', 'b', 'c'], size=n)})
    df_meta = MetaExtractor.extract(df=df_original)
    synthesizer = HighDimSynthesizer(df_meta=df_meta)

    data_imputer = DataImputer(synthesizer=synthesizer)
    assert repr(data_imputer) == f"DataImputer(synthesizer={repr(synthesizer)})"
