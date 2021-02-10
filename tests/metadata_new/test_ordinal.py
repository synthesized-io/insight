import pytest

from synthesized.metadata_new import Bool, OrderedString, Ordinal

from .dataframes import BoolData, OrderedStringData
from .test_nominal import TestNominal as _TestNominal


class TestOrdinal(_TestNominal):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Ordinal:
        meta = Ordinal(name=name)
        return meta

    def test_min(self, extracted_meta, min):
        assert extracted_meta.min == min

    def test_max(self, extracted_meta, max):
        assert extracted_meta.max == max


class TestBool(TestOrdinal, BoolData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Ordinal:
        meta = Bool(name=name)
        return meta


class TestOrderedString(TestOrdinal, OrderedStringData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Ordinal:
        meta = OrderedString(name=name)
        return meta
