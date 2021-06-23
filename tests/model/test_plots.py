import pandas as pd

from synthesized import MetaExtractor
from synthesized.model.factory import ModelFactory


def test_model_plot():

    df = pd.read_csv('data/unittest.csv')
    df_meta = MetaExtractor.extract(df)
    df_model = ModelFactory()(df_meta)
    df_model.fit(df)

    df_model.plot()
