import logging

import pandas as pd

from ..highdim import HighDimSynthesizer


def get_latent_space(df: pd.DataFrame, num_iterations=5_000) -> pd.DataFrame:
    logger = logging.getLogger()
    log_level = logger.level
    logger.setLevel(50)

    with HighDimSynthesizer(df=df, learning_manager=False) as synthesizer:
        synthesizer.learn(df_train=df, num_iterations=num_iterations)
        latent, synthesized = synthesizer.encode(df_encode=df)

    logger.setLevel(log_level)

    return latent
