import logging

import pandas as pd
import pytest

from synthesized.insight.latent import dataset_quality_by_chunk
from tests.utils import progress_bar_testing

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_data_quality_by_chunk():
    data = pd.read_csv('data/credit_with_categoricals.csv').sample(10_000)
    dataset_quality_by_chunk(df=data, n=2, progress_callback=progress_bar_testing)
