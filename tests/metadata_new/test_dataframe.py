import pandas as pd
import pytest

from synthesized.metadata_new import Address, DataFrameMeta, String

from .test_meta import TestMeta as _TestMeta


class TestDataFrame(_TestMeta):

    @pytest.fixture(scope='function')
    def dataframe_orig(self) -> pd.DataFrame:
        return pd.DataFrame({
            'city': pd.Series(['London', 'London', 'London', 'London'], dtype=object),
            'street': pd.Series(['Euston Road', 'Old Kent Road', 'Park Lane', 'Euston Road'], dtype=object),
            'house_number': pd.Series(['1', '2a', '4', '1'], dtype=object)
        })

    @pytest.fixture(scope='function', params=['Empty', 'Full'])
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

        return meta
