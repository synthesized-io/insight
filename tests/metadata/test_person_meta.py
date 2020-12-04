import logging

import pandas as pd
import pytest

from synthesized.metadata import PersonMeta

logger = logging.getLogger(__name__)


@pytest.mark.fast
def test_person_meta_process():

    df = pd.DataFrame({'gender': ['M', 'F']*50, 'first': ['John',]*100, 'last': ['Smith']*100})

    pm = PersonMeta(name='person_0', gender_label='gender', firstname_label='first', lastname_label='last')

    pm.extract(df)
    df2 = pm.preprocess(df)
    pm.postprocess(df2)
