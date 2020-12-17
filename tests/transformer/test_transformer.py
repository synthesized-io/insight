import pandas as pd
import numpy as np
import pytest
import types

from synthesized.metadata_new.meta_builder import MetaExtractor
from synthesized.transformer import Transformer, SequentialTransformer, TransformerFactory, QuantileTransformer, \
    DataFrameTransformer, DateCategoricalTransformer, DateTransformer, CategoricalTransformer, NanTransformer, \
    BinningTransformer, DropColumnTransformer, DTypeTransformer
from synthesized.transformer.exceptions import NonInvertibleTransformError


@pytest.fixture
def df_unittest():
    return pd.read_csv('data/unittest.csv', index_col=0)


@pytest.fixture
def df_unittest_transformers():
    return [CategoricalTransformer(name="SeriousDlqin2yrs", categories=[1, 0]),
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


@pytest.fixture
def sequential_transformer():
    transformer1 = Transformer('transformer1')
    transformer2 = Transformer('transformer2')

    transformer1.transform = types.MethodType(lambda self, x: x, transformer1)
    transformer2.transform = types.MethodType(lambda self, x: x, transformer2)

    transformer1.inverse_transform = types.MethodType(lambda self, x: x, transformer2)
    transformer2.inverse_transform = types.MethodType(lambda self, x: x, transformer2)

    sequential = SequentialTransformer('sequential', transformers=[transformer1, transformer2])

    return sequential, transformer1, transformer2


@pytest.mark.fast
def test_sequential_transformer(sequential_transformer):
    assert sequential_transformer[0][0] == sequential_transformer[1]
    assert sequential_transformer[0][1] == sequential_transformer[2]
    assert sequential_transformer[1] + sequential_transformer[2] == \
        SequentialTransformer('transformer1', transformers=[sequential_transformer[1], sequential_transformer[2]])


@pytest.mark.fast
def test_sequential_transformer_transform(sequential_transformer):
    x = pd.DataFrame({'transformer1': [0], 'transformer2': [0]})
    sequential_transformer[0].fit(x)

    for transformer in sequential_transformer[0]:
        assert transformer._fitted is True


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
        (DropColumnTransformer('x'), pd.DataFrame({'x': [1, 2, 3]})),
        (DTypeTransformer('x'), pd.DataFrame({'x': [1, 2, 3]})),
    ])
def test_transformer(transformer, data):
    transformer.fit(data)
    assert transformer._fitted is True
    try:
        pd.testing.assert_frame_equal(data, transformer.inverse_transform(transformer.transform(data)))
    except NonInvertibleTransformError:
        pass
