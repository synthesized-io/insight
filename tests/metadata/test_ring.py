import pytest

from synthesized.metadata import Ring
from synthesized.metadata.value import Float, IntegerBool

from .dataframes import FloatData, IntBoolData
from .test_scale import TestScale as _TestScale


class TestRing(_TestScale):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Ring:
        meta = Ring(name=name)
        return meta


class TestFloat(TestRing, FloatData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Float:
        meta = Float(name=name)
        return meta


class TestIntegerBool(TestRing, IntBoolData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> IntegerBool:
        meta = IntegerBool(name=name)
        return meta
