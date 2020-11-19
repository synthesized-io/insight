import pandas as pd
import numpy as np
import pytest

from synthesized.metadata.meta import Meta, Nominal, Categorical, Float, Ordinal


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
                            'categories': [],
                            'probabilities': [],
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
    meta['child1'] = Categorical('child1', categories=['A', 'B', 'C'], probabilities=[0.1, 0.2, 0.7])
    meta['child2'] = Meta('child2')
    meta['child2']['child1'] = Float('child1')

    return meta


@pytest.mark.fast
def test_add_child_meta():

    meta = Meta('root')
    child = Nominal('child')
    meta['child'] = child

    assert 'child' in meta
    assert meta.children[0] == child


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
