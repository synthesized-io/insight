import pytest

from synthesized.config import AddressLabels
from synthesized.metadata import DataFrameMeta
from synthesized.metadata.value import Address, String

from .dataframes import AnnotatedDataFrameData, DataFrameData
from .test_meta import TestMeta as _TestMeta


class TestDataFrame(_TestMeta, DataFrameData):

    @pytest.fixture(scope='class', params=['Empty', 'Full'])
    def meta(self, request, name) -> DataFrameMeta:
        meta = DataFrameMeta(name=name)
        if request.param == 'Empty':
            return meta

        number = String('house_number')
        street = String('street')
        city = String('city')

        meta['house_number'] = number
        meta['street'] = street
        meta['city'] = city

        return meta

    @pytest.fixture(scope='function')
    def extracted_meta(self, meta, dataframe):
        df = dataframe.copy()
        meta.extract(df)
        assert dataframe.equals(df)
        yield meta
        meta.__init__(name=meta.name, children=meta.children)


class TestDataFrameAnnotation(_TestMeta, AnnotatedDataFrameData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> DataFrameMeta:
        meta = DataFrameMeta(name=name)

        number = String('house_number')
        street = String('street')
        city = String('city')

        meta['house_number'] = number
        meta['street'] = street
        meta['city'] = city

        meta.annotate(
            Address(
                'address', children=[city, street, number],
                labels=AddressLabels(house_number_label='house_number', street_label='street', city_label='city')
            )
        )
        return meta

    @pytest.fixture(scope='function')
    def extracted_meta(self, meta, dataframe):
        df = dataframe.copy()
        meta.extract(df)
        assert dataframe.equals(df)
        yield meta
        meta.__init__(name=meta.name, children=meta.children, annotations=meta.annotations)