import numpy as np
import pandas as pd
import pytest

from synthesized.metadata_new import Affine, Date, Scale

from .test_ordinal import TestOrdinal as _TestOrdinal


class TestAffine(_TestOrdinal):

    @pytest.fixture(scope='function')
    def meta(self, name) -> Affine:
        meta = Affine(name=name)
        return meta

    def test_unit_meta(self, meta):
        assert isinstance(meta.unit_meta, Scale)

    def test_subtraction(self, extracted_meta):
        if extracted_meta.__class__.__name__ not in ['Affine', 'Scale', 'Ring']:
            assert (extracted_meta.max - extracted_meta.min).dtype == np.dtype(extracted_meta.unit_meta.dtype)


class TestDate(TestAffine):

    @pytest.fixture(scope='function')
    def dataframe_orig(self, name, request) -> pd.DataFrame:
        return pd.DataFrame(
            pd.Series(['2021-01-16', '2021-05-23', '2020-03-01', '2021-02-22'], name=name, dtype='datetime64[ns]')
        )

    @pytest.fixture(scope='function')
    def expanded_dataframe(self, dataframe_orig, name) -> pd.DataFrame:
        return pd.DataFrame({
            f'{name}_dow': pd.Series(['Saturday', 'Sunday', 'Sunday', 'Monday'], name=name, dtype='str'),
            f'{name}_day': pd.Series([16,  23, 1, 22], name=name, dtype='int64'),
            f'{name}_month': pd.Series([1, 5, 3, 2], name=name, dtype='int64'),
            f'{name}_year': pd.Series([2021, 2021, 2020, 2021], name=name, dtype='int64')
        })

    @pytest.fixture(scope='function')
    def meta(self, name) -> Date:
        meta = Date(name=name)
        return meta

    @pytest.fixture(scope='function')
    def categories(self) -> list:
        return [np.datetime64('2020-03-01'), np.datetime64('2021-01-16'),
                np.datetime64('2021-02-22'), np.datetime64('2021-05-23')]

    @pytest.fixture(scope='function')
    def min(self):
        return np.datetime64('2020-03-01')

    @pytest.fixture(scope='function')
    def max(self):
        return np.datetime64('2021-05-23')
