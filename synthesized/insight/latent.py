from typing import Union
import logging

import numpy as np
import pandas as pd

from ..complex.highdim import HighDimSynthesizer, HighDimConfig
from ..metadata import MetaExtractor


def get_latent_space(df: pd.DataFrame, num_iterations=5_000, **kwargs) -> pd.DataFrame:
    logger = logging.getLogger()
    log_level = logger.level
    logger.setLevel(50)

    dp = MetaExtractor.extract(df)
    config = HighDimConfig(learning_manager=False, **kwargs)

    with HighDimSynthesizer(df_meta=dp, config=config) as synthesizer:
        synthesizer.learn(df_train=df, num_iterations=num_iterations)
        df_latent, df_synthesized = synthesizer.encode(df_encode=df)

    logger.setLevel(log_level)

    return df_latent


def get_data_quality(synthesizer: HighDimSynthesizer, df_orig: Union[None, pd.DataFrame],
                     df_new: pd.DataFrame, **kwargs) -> float:
    """Returns a score representing the complexity of the data.

    Internally, this function trains the synthesizer on the new dataframe and uses all the data to calculate a new
    data quality score.

    Args:
        synthesizer: A synthesizer that has (optionally) been train on some set of data.
        df_orig: All of the data that the synthesizer has already been trained on.
        df_new: New data that the synthesizer has not yet been trained on.
        learn_kwargs: kwargs passed to the learn method of the synthesizer.

    """
    with synthesizer as synth:
        synth.learn(df_train=df_new, num_iterations=None, **kwargs)
        df_latent, df_synthesized = synth.encode(df_encode=pd.concat((df_orig, df_new), axis=0))

    return total_latent_space_usage(df_latent, usage_type='mean')


def latent_dimension_usage(df_latent: pd.DataFrame, usage_type: str = 'stddev') -> pd.DataFrame:
    if usage_type == 'stddev':
        ldu = 1.0 - (df_latent.filter(like='s', axis='columns')**2).describe().loc['mean'].round(3).to_numpy()
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
        usage = round(float(np.sum(ldu['usage'].to_numpy())), ndigits=3)
    elif usage_type == 'mean':
        usage = round(float(np.sum(ldu['usage'].to_numpy())), ndigits=3)
    else:
        raise ValueError

    return usage


def density(x, m, v):
    d = (2.0 * np.pi * v)**-0.5 * np.exp(-(x - m)**2 / v)
    return np.sum(d)


def latent_kl_difference(synth: HighDimSynthesizer, df_latent_orig: pd.DataFrame, new_data: pd.DataFrame,
                         num_dims: int = 10):
    dims = latent_dimension_usage(df_latent_orig, 'mean')['dimension'][-10:].to_list()[::-1]
    df_latent, df_syn = synth.encode(new_data)

    mean = df_latent.loc[:, [f'm_{d}' for d in dims]].rename(columns=lambda x: int(x[2:]))
    stddev = df_latent.loc[:, [f's_{d}' for d in dims]].rename(columns=lambda x: int(x[2:]))

    orig_mean = df_latent_orig.loc[:, [f'm_{d}' for d in dims]].rename(columns=lambda x: int(x[2:]))
    orig_stddev = df_latent_orig.loc[:, [f's_{d}' for d in dims]].rename(columns=lambda x: int(x[2:]))

    N = len(df_latent)
    N2 = len(df_latent_orig)
    KL = []

    for dim in dims[:num_dims]:
        m, v = mean[dim].to_numpy()[:N], stddev[dim].to_numpy()[:N] ** 2
        m_orig, v_orig = orig_mean[dim].to_numpy()[:N2], orig_stddev[dim].to_numpy()[:N2] ** 2

        d = 0.02
        X = np.arange(-5, 5, d)
        Y = np.array([density(x, m, v) / N for x in X])
        Y2 = np.array([density(x, m_orig, v_orig) / N2 for x in X])

        KL.append(d * np.sum(Y * np.log(Y / Y2)))

    return np.sum(KL)
