import pandas as pd
import numpy as np
import pytest
import types

from synthesized.metadata.transformer import Transformer, TransformerFactory, QuantileTransformer, DataFrameTransformer, DateCategoricalTransformer, DateTransformer, CategoricalTransformer, NanTransformer
from synthesized.metadata.meta import MetaExtractor


@pytest.fixture
def df_unittest():
    return pd.read_csv('data/unittest.csv', index_col=0)


@pytest.fixture
def nested_transformer():
    parent = Transformer('parent')
    child = Transformer('child')
    child.transform = types.MethodType(lambda self, x: x[self.name], child)
    child.inverse_transform = child.transform
    parent.child = child
    return parent, child


@pytest.mark.fast
def test_add_child_transformer(nested_transformer):
    assert hasattr(nested_transformer[0], 'child')
    assert(nested_transformer[0].transformers[0] == nested_transformer[1])


@pytest.mark.fast
def test_parent_transform(nested_transformer):
    x = pd.DataFrame({'parent': [0], 'child': [0]})
    nested_transformer[0].transform(x)

    for transformer in nested_transformer[0].transformers:
        assert transformer._fitted is True


@pytest.mark.fast
def test_transformer_factory(df_unittest):
    df_meta = MetaExtractor.extract(df_unittest)
    transformer = TransformerFactory().create_transformers(df_meta)
    assert type(transformer) == DataFrameTransformer
    assert len(transformer.transformers) == 12


@pytest.mark.fast
@pytest.mark.parametrize(
    'transformer, data', [
        (QuantileTransformer('x'), pd.DataFrame({'x': np.random.normal(0, 1, size=100)})),
        (CategoricalTransformer('x'), pd.DataFrame({'x': ['A', 'B', 'C']})),
        (DateCategoricalTransformer('x'), pd.DataFrame({'x': ["2013/02/01", "2013/02/03"]})),
        (DateTransformer('x'), pd.DataFrame({'x': ["2013/02/01", "2013/02/03"]})),
        (NanTransformer('x'), pd.DataFrame({'x': [1, 2, np.nan]}))
    ])
def test_transformer(transformer, data):
    transformer.fit(data)
    assert transformer._fitted is True
    pd.testing.assert_frame_equal(data, transformer.inverse_transform(transformer.transform(data)))
