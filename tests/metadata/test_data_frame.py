import logging

import pandas as pd
import pytest

from synthesized.metadata_new import DataFrameMeta, MetaExtractor

logger = logging.getLogger(__name__)


@pytest.mark.fast
def test_load_data_frame():

    df = pd.read_csv('data/unittest.csv')
    df_meta = MetaExtractor.extract(df=df)
    meta_dict = df_meta.get_variables()
    df_meta2 = DataFrameMeta.from_dict(meta_dict)

    assert len(df_meta2.values) == len(df_meta.values)


@pytest.mark.fast
def test_load_data_frame_associations():
    association_dict = {"NumberOfTimes90DaysLate": ["NumberOfTime60-89DaysPastDueNotWorse"]}

    df = pd.read_csv('data/unittest.csv')
    df_meta = MetaExtractor.extract(df=df, associations=association_dict)
    meta_dict = df_meta.get_variables()

    df_meta2 = DataFrameMeta.from_dict(meta_dict)
    assert df_meta2.association_meta is not None
    assert df_meta2.association_meta.binding_mask is not None
