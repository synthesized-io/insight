import numpy as np
import pandas as pd
import pytest

from synthesized.metadata_new import Bool, OrderedString, Ordinal

from .test_nominal import TestNominal as _TestNominal


class TestOrdinal(_TestNominal):

    @pytest.fixture(scope='function', params=["No NaNs", "With NaNs"])
    def dataframe_orig(self, name, request) -> pd.DataFrame:
        if request.param == "No NaNs":
            return pd.DataFrame(pd.Series([True, False, False, True], dtype=bool, name=name))
        else:
            return pd.DataFrame(pd.Series([True, False, np.nan, True], dtype=object, name=name))

    @pytest.fixture(scope='function')
    def meta(self, name) -> Ordinal:
        meta = Ordinal(name=name)
        return meta

    @pytest.fixture(scope='function')
    def categories(self) -> list:
        return [False, True]

    @pytest.fixture(scope='function')
    def min(self):
        return False

    @pytest.fixture(scope='function')
    def max(self):
        return True

    def test_min(self, extracted_meta, min):
        assert extracted_meta.min == min

    def test_max(self, extracted_meta, max):
        assert extracted_meta.max == max


class TestBool(TestOrdinal):

    @pytest.fixture(scope='function')
    def meta(self, name) -> Ordinal:
        meta = Bool(name=name)
        return meta


class TestOrderedString(TestOrdinal):

    @pytest.fixture(scope='function')
    def dataframe_orig(self, name, request) -> pd.DataFrame:
        return pd.DataFrame({name: pd.Categorical(
            ['plum', 'peach', 'pair', 'each', 'plum'],
            dtype=pd.CategoricalDtype(['each', 'peach', 'pair', 'plum'], ordered=True)
        )})

    @pytest.fixture(scope='function')
    def meta(self, name) -> Ordinal:
        meta = OrderedString(name=name)
        return meta

    @pytest.fixture(scope='function')
    def categories(self) -> list:
        return ['each', 'peach', 'pair', 'plum']

    @pytest.fixture(scope='function')
    def min(self):
        return 'each'

    @pytest.fixture(scope='function')
    def max(self):
        return 'plum'
