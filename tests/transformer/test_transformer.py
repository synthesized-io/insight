import pandas as pd
import numpy as np
import pytest
import types

from synthesized.metadata_new.meta_builder import MetaExtractor
from synthesized.transformer import Transformer, TransformerFactory, QuantileTransformer
from synthesized.transformer import DataFrameTransformer, DateCategoricalTransformer, DateTransformer
from synthesized.transformer import CategoricalTransformer, NanTransformer, BinningTransformer, QuantileBinningTransformer
from synthesized.transformer.exceptions import NonInvertibleTransformError


@pytest.fixture
def df_unittest():
    return pd.read_csv('data/unittest.csv', index_col=0)


@pytest.fixture
def df_unittest_transformers():
    return [
        QuantileTransformer(name="SeriousDlqin2yrs", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="RevolvingUtilizationOfUnsecuredLines", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="age", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="NumberOfTime30-59DaysPastDueNotWorse", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="DebtRatio", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="MonthlyIncome", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        NanTransformer(name="MonthlyIncome"),
        QuantileTransformer(name="NumberOfOpenCreditLinesAndLoans", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="NumberOfTimes90DaysLate", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="NumberRealEstateLoansOrLines", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="NumberOfTime60-89DaysPastDueNotWorse", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="NumberOfDependents", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        NanTransformer(name="NumberOfDependents")]


@pytest.mark.fast
def test_transformer_factory(df_unittest, df_unittest_transformers):
    df_meta = MetaExtractor.extract(df_unittest)
    transformer = TransformerFactory().create_transformers(df_meta)
    assert type(transformer) == DataFrameTransformer
    assert transformer._transformers == df_unittest_transformers


@pytest.mark.fast
@pytest.mark.parametrize(
    'transformer, data', [
        (QuantileTransformer('x'), pd.DataFrame({'x': np.random.normal(0, 1, size=100)})),
        (CategoricalTransformer('x'), pd.DataFrame({'x': ['A', 'B', 'C']})),
        (DateCategoricalTransformer('x'), pd.DataFrame({'x': ["2013/02/01", "2013/02/03"]})),
        (DateTransformer('x'), pd.DataFrame({'x': ["2013/02/01", "2013/02/03"]})),
        (NanTransformer('x'), pd.DataFrame({'x': [1, 2, np.nan]})),
        (BinningTransformer('x', bins=10), pd.DataFrame({'x': [1, 2, 3]})),
        (QuantileBinningTransformer('x', quantiles=10), pd.DataFrame({'x': [1, 2, 3]}))

    ])
def test_transformer(transformer, data):
    transformer.fit(data)
    assert transformer._fitted is True
    try:
        pd.testing.assert_frame_equal(data, transformer.inverse_transform(transformer.transform(data)))
    except NonInvertibleTransformError:
        pass
