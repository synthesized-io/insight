import pytest

from synthesized.metadata_new import Address, DataFrameMeta, String

from .dataframes import DataFrameData
from .test_meta import TestMeta as _TestMeta


class TestDataFrame(_TestMeta, DataFrameData):

    @pytest.fixture(scope='class', params=['Empty', 'Full', 'Annotated Address'])
    def meta(self, request, name) -> DataFrameMeta:
        meta = DataFrameMeta(name=name)
        print(request.param)
        if request.param == 'Empty':
            return meta

        number = String('house_number')
        street = String('street')
        city = String('city')

        meta['house_number'] = number
        meta['street'] = street
        meta['city'] = city

        if request.param == 'Annotated Address':
            address = Address('address', house_number_label='house_number', street_label='street', city_label='city')
            meta.apply_annotation(address)

        return meta

    @pytest.fixture(scope='function')
    def extracted_meta(self, meta, dataframe):
        df = dataframe.copy()
        meta.extract(df)
        assert dataframe.equals(df)
        yield meta
        children = meta.children
        meta.__init__(name=meta.name)
        for child in children:
            child.__init__(name=child.name)
            meta[child.name] = child
