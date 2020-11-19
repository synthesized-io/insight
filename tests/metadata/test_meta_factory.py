from dataclasses import asdict

import pandas as pd
import pytest

from synthesized.metadata.meta import Constant, Date, TimeDelta, Nominal, Categorical, Integer, Float, Ordinal, Bool, DataFrameMeta
from synthesized.metadata.meta_builder import _MetaBuilder, MetaFactory


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
def test_extract(nominal_data, ordinal_data, scale_data, date_data):
    meta = MetaFactory().create_meta(nominal_data[0])
    meta = meta.extract(nominal_data[0].to_frame())
    assert meta == nominal_data[1]

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
