import pytest

from synthesized.metadata_new.value import AssociatedCategorical, String

from .dataframes import AssociatedData
from .test_meta import TestMeta as _TestMeta


class TestAssociatedCategorical(_TestMeta, AssociatedData):

    @pytest.fixture(scope='class')
    def meta(self, name, child_names) -> AssociatedCategorical:
        children = [String(name=n) for n in child_names]
        meta = AssociatedCategorical(name=name, children=children)
        return meta

    def test_binding_mask(self, extracted_meta, binding_mask):
        assert (extracted_meta.binding_mask == binding_mask).all()
