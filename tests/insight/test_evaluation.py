import numpy as np
import pandas as pd

from synthesized import MetaExtractor
from synthesized.insight.evaluation import calculate_evaluation_metrics
from synthesized.model.factory import ModelFactory


def test_evaluation_metrics():
    x = np.random.normal(0, 1, 1000)
    y = x + np.random.normal(0, 0.5, 1000)
    df = pd.DataFrame({'x': x, 'y': y})
    df_meta = MetaExtractor.extract(df)
    df_model = ModelFactory()(df_meta)
    vals = calculate_evaluation_metrics(df_orig=df, df_synth=df, df_model=df_model)
