"""Hard Coded Data fixtures for testing the Metas.

This module contains fixtures that provide the hard coded data frames/values to assess the different Metas. Each
Meta class, <ValueMeta[DType]> has some description of the data's representation given by ValueMeta and some
description of the data type, given by DType. Each TestData class in this module is used for a particular DType:

TestData Objects  ->  DType
-------------------|-------------------
StringData         |  'U'
BoolData           |  '?'
OrderedString      |  pd.categorical
DateData           |  'datetime64[ns]'
IntData            |  'i8'
TimeDeltaDay       |  'timedelta64[D]'
FloatData          |  'f8'
IntBoolData        |  'i8'
AddressData        |  addresses
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='class', params=[False, True], ids=['complete', 'with nans'])
def with_nans(request):
    return request.param


class MetaTestData:
    @pytest.fixture(scope='class')
    def dataframe(self, name):
        return pd.DataFrame(columns=[name])

    @pytest.fixture(scope='class')
    def expanded_dataframe(self, dataframe):
        return dataframe.copy()

    @pytest.fixture(scope='class')
    def categories(self) -> list:
        return []

    @pytest.fixture(scope='class')
    def min(self):
        return None

    @pytest.fixture(scope='class')
    def max(self):
        return None


class StringData(MetaTestData):
    @pytest.fixture(scope='class')
    def dataframe(self, name, with_nans) -> pd.DataFrame:
        if with_nans:
            return pd.DataFrame(pd.Series(['a', 'a', 'b', 'c', 'b'], dtype=object, name=name))
        else:
            return pd.DataFrame(pd.Series(['a', np.nan, 'b', 'c', 'b'], dtype=object, name=name))

    @pytest.fixture(scope='class')
    def categories(self) -> list:
        return ['a', 'b', 'c']


class BoolData(MetaTestData):
    @pytest.fixture(scope='class')
    def dataframe(self, name) -> pd.DataFrame:
        return pd.DataFrame(pd.Series([True, False, False, True], dtype=bool, name=name))

    @pytest.fixture(scope='class')
    def categories(self) -> list:
        return [False, True]

    @pytest.fixture(scope='class')
    def min(self) -> bool:
        return False

    @pytest.fixture(scope='class')
    def max(self) -> bool:
        return True


class OrderedStringData(MetaTestData):
    @pytest.fixture(scope='class')
    def dataframe(self, name) -> pd.DataFrame:
        return pd.DataFrame({name: pd.Categorical(
            ['plum', 'peach', 'pair', 'each', 'plum'],
            dtype=pd.CategoricalDtype(['each', 'peach', 'pair', 'plum'], ordered=True)
        )})

    @pytest.fixture(scope='class')
    def categories(self) -> list:
        return ['each', 'peach', 'pair', 'plum']

    @pytest.fixture(scope='class')
    def min(self):
        return 'each'

    @pytest.fixture(scope='class')
    def max(self):
        return 'plum'


class DateData(MetaTestData):
    @pytest.fixture(scope='class')
    def dataframe(self, name) -> pd.DataFrame:
        return pd.DataFrame(
            pd.Series(['2021-01-16', '2021-05-23', '2020-03-01', '2021-02-22'], name=name, dtype='datetime64[ns]')
        )

    @pytest.fixture(scope='class')
    def expanded_dataframe(self, dataframe, name):
        return pd.DataFrame({
            f'{name}_dow': pd.Series(['Saturday', 'Sunday', 'Sunday', 'Monday'], name=name, dtype='str'),
            f'{name}_day': pd.Series([16, 23, 1, 22], name=name, dtype='int64'),
            f'{name}_month': pd.Series([1, 5, 3, 2], name=name, dtype='int64'),
            f'{name}_year': pd.Series([2021, 2021, 2020, 2021], name=name, dtype='int64')
        })

    @pytest.fixture(scope='class')
    def categories(self) -> list:
        return [np.datetime64('2020-03-01'), np.datetime64('2021-01-16'),
                np.datetime64('2021-02-22'), np.datetime64('2021-05-23')]

    @pytest.fixture(scope='class')
    def min(self):
        return np.datetime64('2020-03-01')

    @pytest.fixture(scope='class')
    def max(self):
        return np.datetime64('2021-05-23')


class IntData(MetaTestData):
    @pytest.fixture(scope='class')
    def dataframe(self, name) -> pd.DataFrame:
        return pd.DataFrame(
            pd.Series([-3, 3, 2, 7, 1, 4], name=name, dtype='int64')
        )

    @pytest.fixture(scope='class')
    def categories(self) -> list:
        return [-3, 1, 2, 3, 4, 7]

    @pytest.fixture(scope='class')
    def min(self):
        return np.int64(-3)

    @pytest.fixture(scope='class')
    def max(self):
        return np.int64(7)


class TimeDeltaDayData(MetaTestData):
    @pytest.fixture(scope='class')
    def dataframe(self, name) -> pd.DataFrame:
        return pd.DataFrame(pd.Series(
            [np.timedelta64(3, 'D'), np.timedelta64(1, 'D'), np.timedelta64(3, 'D'), np.timedelta64(4, 'D')],
            name=name,
            dtype='timedelta64[D]'
        ))

    @pytest.fixture(scope='class')
    def categories(self) -> list:
        return [np.timedelta64(1, 'D'), np.timedelta64(3, 'D'), np.timedelta64(4, 'D')]

    @pytest.fixture(scope='class')
    def min(self):
        return np.timedelta64(1, 'D')

    @pytest.fixture(scope='class')
    def max(self):
        return np.timedelta64(4, 'D')


class FloatData(MetaTestData):
    @pytest.fixture(scope='class')
    def dataframe(self, name) -> pd.DataFrame:
        return pd.DataFrame(pd.Series([-3.3, 1.0, 5, 1.0, 13], dtype='float64', name=name))

    @pytest.fixture(scope='class')
    def categories(self) -> list:
        return [-3.3, 1.0, 5, 13]

    @pytest.fixture(scope='class')
    def min(self):
        return np.float64(-3.3)

    @pytest.fixture(scope='class')
    def max(self):
        return np.float64(13)


class IntBoolData(MetaTestData):
    @pytest.fixture(scope='class')
    def dataframe(self, name) -> pd.DataFrame:
        return pd.DataFrame(pd.Series([0, 1, 1, 0], dtype='int64', name=name))

    @pytest.fixture(scope='class')
    def categories(self) -> list:
        return [0, 1]

    @pytest.fixture(scope='class')
    def min(self):
        return np.int64(0)

    @pytest.fixture(scope='class')
    def max(self):
        return np.int64(1)


class AddressData(MetaTestData):
    @pytest.fixture(scope='class')
    def dataframe(self, name) -> pd.DataFrame:
        return pd.DataFrame({
            name: ["1||Blah Drive|Cambridge",
                   "|Housey McHouseface|Placeholder Avenue|London",
                   "42||Test Road|Swansea"]
        }, dtype='string')

    @pytest.fixture(scope='class')
    def expanded_dataframe(self, dataframe):
        return pd.DataFrame({
            "house number": ["1", "", "42"],
            "house name": ["", "Housey McHouseface", ""],
            "street": ["Blah Drive", "Placeholder Avenue", "Test Road"],
            "city": ["Cambridge", "London", "Swansea"],
        })

    @pytest.fixture(scope='class')
    def categories(self) -> list:
        return ["1||Blah Drive|Cambridge",
                "|Housey McHouseface|Placeholder Avenue|London",
                "42||Test Road|Swansea"]


class DataFrameData(MetaTestData):
    @pytest.fixture(scope='class')
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'city': pd.Series(['London', 'London', 'London', 'London'], dtype=object),
            'street': pd.Series(['Euston Road', 'Old Kent Road', 'Park Lane', 'Euston Road'], dtype=object),
            'house_number': pd.Series(['1', '2a', '4', '1'], dtype=object)
        })
