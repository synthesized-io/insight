import pandas as pd
import numpy as np
import pytest

from synthesized.metadata.meta import Meta, Constant, Date, TimeDelta, Nominal, Categorical, Integer, Float, Ordinal, Bool, MetaBuilder, MetaExtractorConfig


@pytest.fixture
def nominal():
    return Nominal(
        'nominal',
        categories=['A', 'B', 'C', 'D', np.nan],
        probabilities=[0.1, 0.2, 0.3, 0.3, 0.1]
    )


@pytest.fixture
def ordinal():
    return Ordinal(
        'ordinal',
        categories=['A', 'B', 'C', 'D']
    )


@pytest.fixture
def nested_dict_meta():
    """Setup a dict object representing a nested Meta structure"""

    return {
        'Meta': {
            'name': 'root',
            'child1': {
                'Categorical': {
                    'name': 'child1',
                    'dtype': None,
                    'categories': ['A', 'B', 'C'],
                    'probabilities': [0.1, 0.2, 0.7],
                    'similarity_based': True
                }
            },
            'child2': {
                'Meta': {
                    'name': 'child2',
                    'child1': {
                        'Float': {
                            'name': 'child1',
                            'dtype': 'float64',
                            'categories': None,
                            'probabilities': None,
                            'similarity_based': True,
                            'min': None,
                            'max': None,
                            'distribution': None,
                            'monotonic': False,
                            'nonnegative': None
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def nested_meta():
    """Setup a Meta object with a nested structure"""

    meta = Meta('root')
    meta.child1 = Categorical('child1', categories=['A', 'B', 'C'], probabilities=[0.1, 0.2, 0.7])
    meta.child2 = Meta('child2')
    meta.child2.child1 = Float('child1')

    return meta


@pytest.mark.fast
def test_add_child_meta():

    meta = Meta('root')
    child = Nominal('child')
    meta.child = child

    assert hasattr(meta, 'child')
    assert(meta.children[0] == child)


@pytest.mark.fast
def test_to_dict(nested_meta, nested_dict_meta):
    d = nested_meta.to_dict()
    assert d == nested_dict_meta


@pytest.mark.fast
def test_from_dict(nested_meta, nested_dict_meta):
    m = Meta.from_dict(nested_dict_meta)
    assert m == nested_meta


@pytest.mark.fast
def test_nominal_probability(nominal):
    assert nominal.probability('A') == 0.1
    assert nominal.probability('B') == 0.2
    assert nominal.probability('C') == 0.3
    assert nominal.probability('D') == 0.3
    assert nominal.probability(np.nan) == 0.1
    assert nominal.probability('F') == 0.0


@pytest.mark.fast
def test_ordinal_less_than(ordinal):
    assert ordinal.less_than('A', 'B') is True
    assert ordinal.less_than('C', 'B') is False

    x = pd.Series(['A', 'D', 'D', 'C', 'B', 'A'])
    assert (ordinal.sort(x) == pd.Series(['A', 'A', 'B', 'C', 'D', 'D'])).all()


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
    builder = MetaBuilder(**MetaExtractorConfig)
    assert type(builder(data)) == meta
