import types

import numpy as np
import pandas as pd
import pytest

from synthesized.config import DateTransformerConfig, QuantileTransformerConfig
from synthesized.metadata_new.meta_builder import MetaExtractor
from synthesized.metadata_new.model import ModelFactory
from synthesized.transformer import (BagOfTransformers, BinningTransformer, CategoricalTransformer,
                                     DataFrameTransformer, DateTransformer, DropColumnTransformer, DTypeTransformer,
                                     NanTransformer, QuantileTransformer, SequentialTransformer, Transformer,
                                     TransformerFactory)
from synthesized.transformer.exceptions import NonInvertibleTransformError


class MockTranformer(Transformer):
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df


@pytest.fixture
def df_credit_with_dates():
    df = pd.read_csv('data/credit_with_categoricals_small.csv')
    df['date'] = [np.datetime64('2017-01-01 00:00:00') + np.timedelta64(np.random.randint(0, 100), 'D') for _ in range(len(df))]
    df['date'] = df['date'].apply(lambda x: x.strftime("%Y/%m/%d"))
    return df


@pytest.fixture
def transformers_credit_with_dates():
    return [
        CategoricalTransformer(name="SeriousDlqin2yrs", categories=[0, 1]),
        QuantileTransformer(name="RevolvingUtilizationOfUnsecuredLines"),
        QuantileTransformer(name="age"),
        CategoricalTransformer(name="NumberOfTime30-59DaysPastDueNotWorse", categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 96, 98]),
        CategoricalTransformer(name="effort", categories=['(0.00649, 0.04]', '(0.00134, 0.00214]', '(-0.001, 0.000309]', '(0.00214, 0.00287]', '(0.04, 12.67]', '(12.67, 3296.64]', '(0.00367, 0.00468]', '(0.00468, 0.00649]', '(0.000309, 0.00134]', '(0.00287, 0.00367]']),
        SequentialTransformer(name="MonthlyIncome", dtypes=None, transformers=[
            NanTransformer(name="MonthlyIncome"),
            QuantileTransformer(name="MonthlyIncome"),
        ]),
        QuantileTransformer(name="NumberOfOpenCreditLinesAndLoans"),
        CategoricalTransformer(name="NumberOfTimes90DaysLate", categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 96, 98]),
        CategoricalTransformer(name="NumberRealEstateLoansOrLines", categories=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17]),
        CategoricalTransformer(name="NumberOfTime60-89DaysPastDueNotWorse", categories=[0, 1, 2, 3, 4, 5, 6, 96, 98]),
        SequentialTransformer(name="NumberOfDependents", dtypes=None, transformers=[
            NanTransformer(name="NumberOfDependents"),
            QuantileTransformer(name="NumberOfDependents"),
        ]),
        DateTransformer(name="date", start_date='2017-01-01')
    ]


@pytest.fixture
def out_columns_credit_with_dates():
    return set(['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'effort', 'MonthlyIncome_nan',
       'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
       'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
       'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents_nan',
       'NumberOfDependents', 'date_month', 'date_day', 'date', 'date_dow',
       'date_hour'])


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


def test_sequential_transformer(sequential_transformer):
    assert sequential_transformer[0][0] == sequential_transformer[1]
    assert sequential_transformer[0][1] == sequential_transformer[2]


def test_sequential_transformer_transform(sequential_transformer):
    x = pd.DataFrame({'transformer1': [0], 'transformer2': [0]})
    sequential_transformer[0].fit(x)

    for transformer in sequential_transformer[0]:
        assert transformer._fitted is True


def test_transformer_factory(df_credit_with_dates, transformers_credit_with_dates, out_columns_credit_with_dates):
    df_meta = MetaExtractor.extract(df_credit_with_dates)
    df_model = ModelFactory().create_df_model_meta(df_meta)
    transformer = TransformerFactory().create_transformers(df_model)
    df_transformer = DataFrameTransformer.from_meta(df_model)

    assert type(transformer) == DataFrameTransformer
    assert transformer._transformers == transformers_credit_with_dates

    df_transformer.fit(df_credit_with_dates)
    out_columns = set(df_transformer.transform(df_credit_with_dates).columns)
    assert out_columns == out_columns_credit_with_dates


@pytest.mark.parametrize(
    'transformer, data', [
        (QuantileTransformer('x', config=QuantileTransformerConfig(noise=None)), pd.DataFrame({'x': np.random.normal(0, 1, size=100)}).astype(np.float32)),
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


def test_date_transformer():
    config = DateTransformerConfig(noise=None)
    transformer = DateTransformer(name="date", config=config)

    n = 5000
    df = pd.DataFrame([np.datetime64('2017-01-01 00:00:00') + np.random.randint(1000, 1_000_000) for _ in range(n)],
                      columns=['date'])
    df['date'] = df['date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    transformer.fit(df)
    assert transformer._fitted is True
    df_t = transformer.transform(df.copy())
    pd.testing.assert_frame_equal(df, transformer.inverse_transform(df_t))


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
