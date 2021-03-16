import logging

import pytest

from synthesized.metadata_new import Meta

from .dataframes import MetaTestData

logger = logging.getLogger(__name__)


def reset_meta(meta: Meta):
    children = meta.children
    if len(children) > 0:
        for child in children:
            reset_meta(child)
        meta.__init__(name=meta.name, children=children)  # type: ignore
    else:
        meta.__init__(name=meta.name)  # type: ignore


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
    def meta(self, name) -> Meta:
        meta: Meta = Meta(name=name)
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
        reset_meta(meta)

    def test_extract(self, extracted_meta, dataframe):
        assert extracted_meta._extracted
        for child in extracted_meta.children:
            assert child._extracted

    @pytest.fixture(scope='class')
    def meta_expanded_dataframe(self, meta, dataframe):
        df = dataframe.copy()
        meta.convert_df_for_children(df)
        return df

    @pytest.fixture(scope='class')
    def collapsed_dataframe(self, meta, meta_expanded_dataframe):
        df = meta_expanded_dataframe.copy()
        meta.revert_df_from_children(df)
        return df

    def test_expand(self, meta, meta_expanded_dataframe, expanded_dataframe):

        for child in meta.children:
            assert child.name in meta_expanded_dataframe.columns
        assert expanded_dataframe.equals(meta_expanded_dataframe)
        assert expanded_dataframe.columns.equals(meta_expanded_dataframe.columns)

    def test_collapse(self, collapsed_dataframe, dataframe):
        columns = sorted(dataframe.columns)
        assert columns == sorted(dataframe.columns)
        assert dataframe[columns].equals(collapsed_dataframe[columns])

    def test_serialisation(self, extracted_meta):
        dct = extracted_meta.to_dict()
        new_meta = extracted_meta.from_dict(dct)
        assert new_meta == extracted_meta


def test_meta_equals():

    assert Meta(name='a') == Meta(name='a')
    assert Meta(name='a') != Meta(name='b')
