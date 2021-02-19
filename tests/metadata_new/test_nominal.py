import logging

import pytest

from synthesized.config import PersonLabels
from synthesized.metadata_new import Address, Bank, Nominal, Person, String

from .dataframes import AddressData, BankData, PersonData, StringData
from .test_meta import TestMeta as _TestMeta

logger = logging.getLogger(__name__)


class TestNominal(_TestMeta):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Nominal:
        meta = Nominal(name=name)
        return meta

    def test_categories(self, extracted_meta, categories):
        assert extracted_meta.categories == categories


class TestString(TestNominal, StringData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Nominal:
        meta = String(name=name)
        return meta


class TestAddress(TestNominal, AddressData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Address:
        meta = Address(
            name=name, street_label='street', city_label='city', house_name_label='house name',
            house_number_label='house number'
        )
        return meta

    @pytest.fixture(scope='function')
    def extracted_meta(self, meta, dataframe):
        df = dataframe.copy()
        meta.extract(df)
        assert dataframe.equals(df)
        yield meta
        meta.__init__(
            name=meta.name, street_label='street', city_label='city', house_name_label='house name',
            house_number_label='house number'
        )


class TestPerson(TestNominal, PersonData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Person:
        meta = Person(
            name=name, labels=PersonLabels(firstname_label='firstname', lastname_label='lastname')
        )
        return meta

    @pytest.fixture(scope='function')
    def extracted_meta(self, meta, dataframe):
        df = dataframe.copy()
        meta.extract(df)
        assert dataframe.equals(df)
        yield meta
        meta.__init__(
            name=meta.name, labels=PersonLabels(firstname_label="first_name", lastname_label="last_name")
        )


class TestBank(TestNominal, BankData):

    @pytest.fixture(scope='class')
    def meta(self, name) -> Bank:
        meta = Bank(
            name=name, bic_label='bic', sort_code_label='sort_code', account_label='account'
        )
        return meta

    @pytest.fixture(scope='function')
    def extracted_meta(self, meta, dataframe):
        df = dataframe.copy()
        meta.extract(df)
        assert dataframe.equals(df)
        yield meta
        meta.__init__(
            name=meta.name, bic_label='bic', sort_code_label='sort_code', account_label='account'
        )
