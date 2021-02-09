import numpy as np
import pandas as pd
import pytest

from synthesized.metadata_new import Integer, Scale, TimeDeltaDay

from .test_affine import TestAffine as _TestAffine


class TestScale(_TestAffine):

    @pytest.fixture(scope='function')
    def meta(self, name) -> Scale:
        meta = Scale(name=name)
        return meta

    def test_scale_division(self, extracted_meta):
        if extracted_meta.__class__.__name__ not in ['Scale', 'Ring']:
            assert (extracted_meta.max / extracted_meta.precision).dtype == np.dtype('float64')


class TestInteger(TestScale):

    @pytest.fixture(scope='function')
    def dataframe_orig(self, name, request) -> pd.DataFrame:
        return pd.DataFrame(
            pd.Series([-3, 3, 2, 7, 1, 4], name=name, dtype='int64')
        )

    @pytest.fixture(scope='function')
    def meta(self, name) -> Integer:
        meta = Integer(name=name)
        return meta

    @pytest.fixture(scope='function')
    def categories(self) -> list:
        return [-3, 1, 2, 3, 4, 7]

    @pytest.fixture(scope='function')
    def min(self):
        return np.int64(-3)

    @pytest.fixture(scope='function')
    def max(self):
        return np.int64(7)


class TestTimeDeltaDay(TestScale):

    @pytest.fixture(scope='function')
    def dataframe_orig(self, name, request) -> pd.DataFrame:
        return pd.DataFrame(
            pd.Series([np.timedelta64(3, 'D'), np.timedelta64(1, 'D'), np.timedelta64(4, 'D')], name=name, dtype='timedelta64[D]')
        )

    @pytest.fixture(scope='function')
    def meta(self, name) -> TimeDeltaDay:
        meta = TimeDeltaDay(name=name)
        return meta

    @pytest.fixture(scope='function')
    def categories(self) -> list:
        return [np.timedelta64(1, 'D'), np.timedelta64(3, 'D'), np.timedelta64(4, 'D')]

    @pytest.fixture(scope='function')
    def min(self):
        return np.timedelta64(1, 'D')

    @pytest.fixture(scope='function')
    def max(self):
        return np.timedelta64(4, 'D')
