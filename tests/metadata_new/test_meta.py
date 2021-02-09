import logging

import pandas as pd
import pytest

from synthesized.metadata_new import Meta

logger = logging.getLogger(__name__)


class TestMeta:
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
    @pytest.fixture(scope='function')
    def name(self) -> str:
        return 'x'

    @pytest.fixture(scope='function')
    def meta(self, name) -> 'Meta':
        meta = Meta(name=name)
        return meta

    @pytest.fixture(scope='function')
    def dataframe_orig(self, name) -> pd.DataFrame:
        return pd.DataFrame(pd.Series(name=name, dtype=object))

    @pytest.fixture(scope='function')
    def dataframe(self, dataframe_orig) -> pd.DataFrame:
        return dataframe_orig.copy()

    def test_name(self, meta, name):
        assert meta.name == name
        for child_name, child in meta.items():
            assert child.name == child_name

    @pytest.fixture(scope='function')
    def extracted_meta(self, meta, dataframe):
        meta.extract(dataframe)
        return meta

    def test_extract(self, extracted_meta, dataframe, dataframe_orig):
        assert extracted_meta._extracted
        for child in extracted_meta.children:
            assert child._extracted

        assert dataframe.equals(dataframe_orig)

    @pytest.fixture(scope='function')
    def expand_dataframe(self, meta, dataframe):
        meta.expand(dataframe)

    @pytest.fixture(scope='function')
    def collapse_dataframe(self, meta, expand_dataframe, dataframe):
        meta.collapse(dataframe)

    def test_expand(self, meta, expand_dataframe, dataframe):
        for child in meta.children:
            assert child.name in dataframe.columns

    def test_collapse(self, collapse_dataframe, dataframe, dataframe_orig):
        assert dataframe.equals(dataframe_orig)

    def test_serialisation(self, extracted_meta):
        dct = extracted_meta.to_dict()
        new_meta = extracted_meta.from_dict(dct)
        assert new_meta == extracted_meta
