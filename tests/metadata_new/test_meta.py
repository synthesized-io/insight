import logging

import pytest

from synthesized.metadata_new import Meta

from .dataframes import MetaTestData

logger = logging.getLogger(__name__)


class TestMeta(MetaTestData):
    """Test set for testing the base Meta class.

    To Define in Subclasses:
        - meta
        - dataframe_orig

    Tests:
        - instantiation
        - extraction
        - expand/collapse
        - serialisation
    """
    @pytest.fixture(scope='class')
    def meta(self, name) -> 'Meta':
        meta = Meta(name=name)
        return meta

    def test_name(self, meta, name):
        assert meta.name == name
        for child_name, child in meta.items():
            assert child.name == child_name

    @pytest.fixture(scope='function')
    def extracted_meta(self, meta, dataframe):
        df = dataframe.copy()
        meta.extract(df)
        assert dataframe.equals(df)
        yield meta
        meta.__init__(name=meta.name)

    def test_extract(self, extracted_meta, dataframe):
        assert extracted_meta._extracted
        for child in extracted_meta.children:
            assert child._extracted

    @pytest.fixture(scope='class')
    def meta_expanded_dataframe(self, meta, dataframe):
        df = dataframe.copy()
        meta.expand(df)
        return df

    @pytest.fixture(scope='class')
    def collapsed_dataframe(self, meta, meta_expanded_dataframe):
        df = meta_expanded_dataframe.copy()
        meta.collapse(df)
        return df

    def test_expand(self, meta, meta_expanded_dataframe, expanded_dataframe):

        for child in meta.children:
            assert child.name in meta_expanded_dataframe.columns

        assert expanded_dataframe.equals(meta_expanded_dataframe)

    def test_collapse(self, collapsed_dataframe, dataframe):
        assert dataframe.equals(collapsed_dataframe)

    def test_serialisation(self, extracted_meta):
        dct = extracted_meta.to_dict()
        new_meta = extracted_meta.from_dict(dct)
        assert new_meta == extracted_meta
