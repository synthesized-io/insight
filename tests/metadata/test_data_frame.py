import logging

import pandas as pd
import pytest

from synthesized.metadata_new import DataFrameMeta, MetaExtractor

logger = logging.getLogger(__name__)


def test_load_data_frame():

    df = pd.read_csv('data/unittest.csv')
    df_meta = MetaExtractor.extract(df=df)
    meta_dict = df_meta.to_dict()
    df_meta2 = DataFrameMeta.from_dict(meta_dict)

    assert len(df_meta2) == len(df_meta)
    assert [name1 == name2 for name1, name2 in zip(df_meta, df_meta2)]


@pytest.mark.skip(reason="Currently in development")
def test_load_data_frame_associations():
    association_dict = {"NumberOfTimes90DaysLate": ["NumberOfTime60-89DaysPastDueNotWorse"]}

    df = pd.read_csv('data/unittest.csv')
    df_meta = MetaExtractor.extract(df=df, associations=association_dict)
    meta_dict = df_meta.to_dict()

    df_meta2 = DataFrameMeta.from_dict(meta_dict)
    assert df_meta2.association_meta is not None
    assert df_meta2.association_meta.binding_mask is not None
