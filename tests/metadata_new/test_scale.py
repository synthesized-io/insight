import numpy as np
import pytest

from synthesized.metadata_new import Scale
from synthesized.metadata_new.value import Integer, TimeDeltaDay

from .dataframes import IntData, TimeDeltaDayData
from .test_affine import TestAffine as _TestAffine


class TestScale(_TestAffine):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Scale:
        meta = Scale(name=name)
        return meta

    def test_scale_division(self, extracted_meta):
        if extracted_meta.__class__.__name__ not in ['Scale', 'Ring']:
            assert (extracted_meta.max / extracted_meta.precision).dtype == np.dtype('float64')


class TestInteger(TestScale, IntData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Integer:
        meta = Integer(name=name)
        return meta


class TestTimeDeltaDay(TestScale, TimeDeltaDayData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> TimeDeltaDay:
        meta = TimeDeltaDay(name=name)
        return meta
