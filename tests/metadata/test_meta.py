import pandas as pd
import numpy as np
import pytest

from synthesized.metadata.meta import Meta, Date, Nominal, Categorical, Integer, Float, Ordinal


@pytest.fixture
def nominal_data():
    return pd.Series([1.2, 'A', 2, np.nan] * 25, name='nominal'), \
        Nominal(name='nominal', categories=[np.nan, 1.2, 2, 'A'], probabilities=[0.25, 0.25, 0.25, 0.25], dtype='object')


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
