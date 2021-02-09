import logging

import pandas as pd
import pytest

from synthesized.metadata_new import Address, Nominal, String

from .test_meta import TestMeta as _TestMeta

logger = logging.getLogger(__name__)


class TestNominal(_TestMeta):

    @pytest.fixture(scope='function')
    def dataframe_orig(self, name) -> pd.DataFrame:
        return pd.DataFrame(pd.Series(['a', 'a', 'b', 'c', 'b'], dtype=object, name=name))

    @pytest.fixture(scope='function')
    def categories(self) -> list:
        return ['a', 'b', 'c']

    @pytest.fixture(scope='function')
    def meta(self, name) -> Nominal:
        meta = Nominal(name=name)
        return meta

    def test_categories(self, extracted_meta, categories):
        assert extracted_meta.categories == categories


class TestString(TestNominal):

    @pytest.fixture(scope='function')
    def meta(self, name, categories) -> Nominal:
        meta = String(name=name, categories=categories)
        return meta


class TestAddress(TestNominal):

    @pytest.fixture(scope='function')
    def dataframe_orig(self, name) -> pd.DataFrame:
        return pd.DataFrame({
            name: ["1||Blah Drive|Cambridge",
                   "|Housey McHouseface|Placeholder Avenue|London",
                   "42||Test Road|Swansea"]
        })

    @pytest.fixture(scope='function')
    def expanded_dataframe(self, dataframe_orig):
        return pd.DataFrame({
            "house number": ["1", "", "42"],
            "house name": ["", "Housey McHouseface", ""],
            "street": ["Blah Drive", "Placeholder Avenue", "Test Road"],
            "city": ["Cambridge", "London", "Swansea"],
        })

    @pytest.fixture(scope='function')
    def categories(self) -> list:
        return ["1||Blah Drive|Cambridge",
                "|Housey McHouseface|Placeholder Avenue|London",
                "42||Test Road|Swansea"]

    @pytest.fixture(scope='function')
    def meta(self, name) -> Address:
        meta = Address(
            name=name, street_label='street', city_label='city', house_name_label='house name',
            house_number_label='house number'
        )
        return meta
