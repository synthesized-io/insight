from dataclasses import asdict

import pandas as pd
import numpy as np
import pytest

from synthesized.metadata.meta import Constant, Date, TimeDelta, Nominal, Categorical, Integer, Float, Ordinal, Bool, DataFrameMeta
from synthesized.metadata.meta_builder import _MetaBuilder, MetaFactory


@pytest.fixture
def df_unittest_column_meta():
    return {'SeriousDlqin2yrs': Categorical(name='SeriousDlqin2yrs', similarity_based=False),
            'RevolvingUtilizationOfUnsecuredLines': Float(name='RevolvingUtilizationOfUnsecuredLines'),
            'age': Integer(name='age'),
            'NumberOfTime30-59DaysPastDueNotWorse': Categorical(name='NumberOfTime30-59DaysPastDueNotWorse'),
            'DebtRatio': Float(name='DebtRatio'),
            'MonthlyIncome': Integer(name='MonthlyIncome'),
            'NumberOfOpenCreditLinesAndLoans': Integer(name='NumberOfOpenCreditLinesAndLoans'),
            'NumberOfTimes90DaysLate': Categorical(name='NumberOfTimes90DaysLate'),
            'NumberRealEstateLoansOrLines': Categorical(name='NumberRealEstateLoansOrLines'),
            'NumberOfTime60-89DaysPastDueNotWorse': Categorical(name='NumberOfTime60-89DaysPastDueNotWorse'),
            'NumberOfDependents': Categorical(name='NumberOfDependents')}


@pytest.fixture
def df_unittest():
    return pd.read_csv('data/unittest.csv', index_col=0)


@pytest.fixture
def ordinal_data():
    categories = ['extra mild', 'mild', 'medium', 'hot', 'extra hot']
    x = pd.Series(categories * 20, name='ordinal', dtype=pd.CategoricalDtype(categories=categories, ordered=True))
    x = x.cat.as_ordered()
    return x, Ordinal(name='ordinal', min='extra mild', max='extra hot', categories=categories, probabilities=len(categories) * [1 / len(categories)], dtype='category')


@pytest.fixture
def scale_data():
    x = pd.Series(np.random.normal(loc=0, scale=1, size=100), name='scale')
    return x, Float(name='scale', min=x.min(), max=x.max(), monotonic=False, nonnegative=False)


@pytest.fixture
def date_data():
    x = pd.Series(pd.date_range("01/01/1993", "01/01/2000", periods=100).strftime("%d/%m/%Y"), name='date')
    return x, Date(name='date', date_format="%d/%m/%Y", min=pd.Timestamp(year=1993, month=1, day=1),
                   monotonic=True, max=pd.Timestamp(year=2000, month=1, day=1))


data_meta = [
    (pd.Series(['1.2', 2, 'a']), Nominal),
    (pd.Series(['1.2', '1.2']), Constant),
    (pd.Series([True, False]), Bool),
    (pd.Series(['A', 'B']), Categorical),
    (pd.Series(['1', '2', '3']), Categorical),
    (pd.Series(['A', 'B', 'C'], dtype=pd.CategoricalDtype(categories=['A', 'B', 'C'], ordered=False)), Categorical),
    (pd.Series(['A', 'B', 'C'], dtype=pd.CategoricalDtype(categories=['A', 'B', 'C'], ordered=True)), Ordinal),
    (pd.Series([1, 2]), Categorical),
    (pd.Series([1.0, 2.0]), Categorical),
    (pd.Series(['1', 1, 1.0]), Categorical),
    (pd.Series(['1', '2.0', '3']), Categorical),
    (pd.Series(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']), Integer),
    (pd.Series([1.2, 2.3]), Float),
    (pd.Series([1, 2.3]), Float),
    (pd.Series(['1', '2.3', '3.1']), Float),
    (pd.Series(['1.2', 2.3]), Float),
    (pd.Series(["2013/02/01", "2013/02/03"]), Date),
    (pd.Series([226, 232], dtype='timedelta64[ns]'), TimeDelta)
]


@pytest.mark.fast
@pytest.mark.parametrize(
    "data, meta", data_meta
)
def test_default_builder(data, meta):
    builder = _MetaBuilder(**asdict(MetaFactory.default_config()))
    assert type(builder(data)) == meta


@pytest.mark.fast
def test_meta_factory(df_unittest, df_unittest_column_meta):
    df_meta = MetaFactory()(df_unittest)
    assert type(df_meta) == DataFrameMeta
    assert len(df_meta.children) == 11
    assert df_meta.column_meta == df_unittest_column_meta


@pytest.mark.fast
def test_extract(ordinal_data, scale_data, date_data):
    meta = MetaFactory().create_meta(ordinal_data[0])
    meta = meta.extract(ordinal_data[0].to_frame())
    assert meta == ordinal_data[1]

    meta = MetaFactory().create_meta(scale_data[0])
    meta = meta.extract(scale_data[0].to_frame())
    meta.categories = []
    meta.probabilities = []
    assert meta == scale_data[1]

    meta = MetaFactory().create_meta(date_data[0])
    meta = meta.extract(date_data[0].to_frame())
    meta.categories = []
    meta.probabilities = []
    assert meta == date_data[1]
