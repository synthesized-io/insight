import logging

import pandas as pd
import pytest

from synthesized.insight.latent import dataset_quality_by_chunk
from synthesized.testing import testing_progress_bar

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_data_quality_by_chunk():
    data = pd.read_csv('data/credit_with_categoricals.csv').sample(10_000)
    dataset_quality_by_chunk(df=data, n=2, progress_callback=testing_progress_bar)
