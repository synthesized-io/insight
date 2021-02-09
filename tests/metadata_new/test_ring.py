import numpy as np
import pandas as pd
import pytest

from synthesized.metadata_new import Float, IntegerBool, Ring

from .test_scale import TestScale as _TestScale


class TestRing(_TestScale):

    @pytest.fixture(scope='function')
    def meta(self, name) -> Ring:
        meta = Ring(name=name)
        return meta


class TestFloat(TestRing):

    @pytest.fixture(scope='function', params=["No NaNs", "With NaNs"])
    def dataframe_orig(self, name, request) -> pd.DataFrame:
        if request.param == "No NaNs":
            return pd.DataFrame(pd.Series([-3.3, 1.0, 5, 13], dtype='float64', name=name))
        else:
            return pd.DataFrame(pd.Series([-3.3, 1.0,  5, np.nan, 13], dtype='float64', name=name))

    @pytest.fixture(scope='function')
    def meta(self, name) -> Float:
        meta = Float(name=name)
        return meta

    @pytest.fixture(scope='function')
    def categories(self) -> list:
        return [-3.3, 1.0, 5, 13]

    @pytest.fixture(scope='function')
    def min(self):
        return np.float64(-3.3)

    @pytest.fixture(scope='function')
    def max(self):
        return np.float64(13)


class TestIntegerBool(TestRing):

    @pytest.fixture(scope='function', params=["No NaNs", "With NaNs"])
    def dataframe_orig(self, name, request) -> pd.DataFrame:
        if request.param == "No NaNs":
            return pd.DataFrame(pd.Series([0, 1, 1, 0], dtype='int64', name=name))
        else:
            return pd.DataFrame(pd.Series([0, 1,  1, np.nan], dtype=object, name=name))

    @pytest.fixture(scope='function')
    def meta(self, name) -> IntegerBool:
        meta = IntegerBool(name=name)
        return meta

    @pytest.fixture(scope='function')
    def categories(self) -> list:
        return [0, 1]

    @pytest.fixture(scope='function')
    def min(self):
        return np.int64(0)

    @pytest.fixture(scope='function')
    def max(self):
        return np.int64(1)
