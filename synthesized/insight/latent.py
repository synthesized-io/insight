import logging

import numpy as np
import pandas as pd

from ..highdim import HighDimSynthesizer


def get_latent_space(df: pd.DataFrame, num_iterations=5_000) -> pd.DataFrame:
    logger = logging.getLogger()
    log_level = logger.level
    logger.setLevel(50)

    with HighDimSynthesizer(df=df, learning_manager=False) as synthesizer:
        synthesizer.learn(df_train=df, num_iterations=num_iterations)
        df_latent, df_synthesized = synthesizer.encode(df_encode=df)

    logger.setLevel(log_level)

    return df_latent


def latent_dimension_usage(df_latent: pd.DataFrame, usage_type: str = 'stddev') -> pd.DataFrame:
    if usage_type == 'stddev':
        ldu = df_latent.filter(like='s', axis='columns').describe().loc['mean'].round(3).to_numpy()
    elif usage_type == 'mean':
        ldu = df_latent.filter(like='m', axis='columns').describe().loc['std'].round(3).to_numpy()
    else:
        raise ValueError

    ldu = pd.DataFrame(dict(
        dimension=np.arange(len(ldu)), usage=ldu
    ))

    return ldu.sort_values(by='usage').reset_index(drop=True)


def total_latent_space_usage(df_latent: pd.DataFrame, usage_type: str = 'stddev') -> float:
    ldu = latent_dimension_usage(df_latent=df_latent, usage_type=usage_type)

    if usage_type == 'stddev':
        usage = round(float(np.sum(1.0-ldu['usage'].to_numpy())), ndigits=3)
    elif usage_type == 'mean':
        usage = round(float(np.sum(ldu['usage'].to_numpy())), ndigits=3)
    else:
        raise ValueError

    return usage
