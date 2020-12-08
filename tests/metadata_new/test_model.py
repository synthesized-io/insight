import logging

import pytest
import pandas as pd

from synthesized.metadata_new import MetaExtractor
from synthesized.metadata_new.model import Histogram, KernelDensityEstimate

logger = logging.getLogger(__name__)


@pytest.mark.fast
def test_load_data_frame():
    df = pd.read_csv('data/unittest.csv')
    meta = MetaExtractor.extract(df)

    col_meta = meta.column_meta

    for col in df.columns:
        logger.info(col_meta[col].to_dict())
