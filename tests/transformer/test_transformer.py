import types

import numpy as np
import pandas as pd
import pytest

from synthesized.metadata_new.meta_builder import MetaExtractor
from synthesized.transformer import (BagOfTransformers, BinningTransformer, CategoricalTransformer,
                                     DataFrameTransformer, DateCategoricalTransformer, DateToNumericTransformer,
                                     DateTransformer, DropColumnTransformer, DTypeTransformer, NanTransformer,
                                     QuantileTransformer, SequentialTransformer, Transformer, TransformerFactory)
from synthesized.transformer.exceptions import NonInvertibleTransformError


class MockTranformer(Transformer):
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df


@pytest.fixture
def df_credit_with_dates():
    df = pd.read_csv('data/credit_with_categoricals_small.csv')
    df['date'] = [np.datetime64('2017-01-01 00:00:00') + np.random.randint(1000, 1_000_000) for _ in range(len(df))]
    df['date'] = df['date'].apply(lambda x: x.strftime("%Y/%m/%d"))
    return df


@pytest.fixture
def df_credit_with_dates_transformers():
    return [
        CategoricalTransformer(name="SeriousDlqin2yrs", categories=[0, 1]),
        QuantileTransformer(name="RevolvingUtilizationOfUnsecuredLines", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="age", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="NumberOfTime30-59DaysPastDueNotWorse", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        CategoricalTransformer(name="effort", categories=['(0.00649, 0.04]', '(0.00134, 0.00214]', '(-0.001, 0.000309]', '(0.00214, 0.00287]', '(0.04, 12.67]', '(12.67, 3296.64]', '(0.00367, 0.00468]', '(0.00468, 0.00649]', '(0.000309, 0.00134]', '(0.00287, 0.00367]']),
        SequentialTransformer(name="MonthlyIncome", dtypes=None, transformers=[
            NanTransformer(name="MonthlyIncome"),
            QuantileTransformer(name="MonthlyIncome", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        ]),
        QuantileTransformer(name="NumberOfOpenCreditLinesAndLoans", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="NumberOfTimes90DaysLate", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        QuantileTransformer(name="NumberRealEstateLoansOrLines", n_quantiles=1000, output_distribution="normal", noise=1e-07),
            QuantileTransformer(name="NumberOfTime60-89DaysPastDueNotWorse", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        SequentialTransformer(name="NumberOfDependents", dtypes=None, transformers=[
            NanTransformer(name="NumberOfDependents"),
            QuantileTransformer(name="NumberOfDependents", n_quantiles=1000, output_distribution="normal", noise=1e-07),
        ]),
        DateTransformer(name="date", date_format="%Y/%m/%d", unit="days", start_date='2017-01-01', n_quantiles=1000, output_distribution="normal", noise=1e-07)
    ]


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


@pytest.mark.fast
def test_sequential_transformer_transform(sequential_transformer):
    x = pd.DataFrame({'transformer1': [0], 'transformer2': [0]})
    sequential_transformer[0].fit(x)

    for transformer in sequential_transformer[0]:
        assert transformer._fitted is True


@pytest.mark.fast
def test_transformer_factory(df_credit_with_dates, df_credit_with_dates_transformers):
    df_meta = MetaExtractor.extract(df_credit_with_dates)
    transformer = TransformerFactory().create_transformers(df_meta)
    df_transformer = DataFrameTransformer.from_meta(df_meta)

    assert type(transformer) == DataFrameTransformer
    assert transformer._transformers == df_credit_with_dates_transformers


@pytest.mark.fast
@pytest.mark.parametrize(
    'transformer, data', [
        (QuantileTransformer('x', noise=None), pd.DataFrame({'x': np.random.normal(0, 1, size=100)}).astype(np.float32)),
        (CategoricalTransformer('x'), pd.DataFrame({'x': ['A', 'B', 'C']})),
        (CategoricalTransformer('x'), pd.DataFrame({'x': [0, 1, np.nan]})),
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


@pytest.mark.fast
def test_date_transformer():
    date_format = "%Y/%m/%d"
    transformer = DateTransformer(name="date", date_format=date_format, noise=None)

    n = 5000
    df = pd.DataFrame([np.datetime64('2017-01-01 00:00:00') + np.random.randint(1000, 1_000_000) for _ in range(n)],
                      columns=['date'])
    df['date'] = df['date'].apply(lambda x: x.strftime("%Y/%m/%d"))
    transformer.fit(df)
    assert transformer._fitted is True
    df_t = transformer.transform(df.copy())
    pd.testing.assert_frame_equal(df, transformer.inverse_transform(df_t))


@pytest.mark.fast
def test_complex_sequence_of_transformers():

    bag_of_transformers = BagOfTransformers('bot', transformers=[
        MockTranformer('x1'),
        SequentialTransformer('seq', transformers=[
            MockTranformer('x21'),
            BagOfTransformers('bot', transformers=[MockTranformer('x22'), MockTranformer('x22')]),
        ]),
        MockTranformer('x3'),
    ])

    n = 1000
    df = pd.DataFrame({'x1': np.random.normal(size=n), 'x21': np.random.normal(size=n), 'x22': np.random.normal(size=n), 'x3': np.random.normal(size=n)})
    bag_of_transformers.fit(df)
    bag_of_transformers.inverse_transform(bag_of_transformers.transform(df))
