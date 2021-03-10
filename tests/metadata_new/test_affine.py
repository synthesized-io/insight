import numpy as np
import pytest

from synthesized.metadata_new import Affine, Scale
from synthesized.metadata_new.value import DateTime

from .dataframes import DateData
from .test_ordinal import TestOrdinal as _TestOrdinal


class TestAffine(_TestOrdinal):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Affine:
        meta = Affine(name=name)
        return meta

    def test_unit_meta(self, meta):
        assert isinstance(meta.unit_meta, Scale)

    def test_subtraction(self, extracted_meta):
        if extracted_meta.__class__.__name__ not in ['Affine', 'Scale', 'Ring']:
            assert (extracted_meta.max - extracted_meta.min).dtype == np.dtype(extracted_meta.unit_meta.dtype)


class TestDate(TestAffine, DateData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> DateTime:
        meta = DateTime(name=name)
        return meta
