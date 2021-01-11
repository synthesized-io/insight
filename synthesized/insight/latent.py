from typing import Callable, Optional, Union
import logging

import numpy as np
import pandas as pd

from ..complex.highdim import HighDimSynthesizer, HighDimConfig
from ..metadata_new import MetaExtractor


def get_latent_space(df: pd.DataFrame, num_iterations=5_000, **kwargs) -> pd.DataFrame:
    """Generates a latent space representation for a given DataFrame using the HighDimSynthesizer.

    Args:
        df: The dataframe to generate the latent space for.
        num_iterations: The number of iterations spent refining the latent space.
        **kwargs: Keyword arguments to pass to the HighDimConfig

    Returns:
        The latent space as a pandas dataframe.

    """
    logger = logging.getLogger()
    log_level = logger.level
    logger.setLevel(50)

    df_meta = MetaExtractor.extract(df)
    config = HighDimConfig(learning_manager=False, **kwargs)

    with HighDimSynthesizer(df_meta=df_meta, config=config) as synthesizer:
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
        **kwargs: kwargs passed to the learn method of the synthesizer.

    """
    with synthesizer as synth:
        synth.learn(df_train=df_new, num_iterations=None, **kwargs)
        df_latent, df_synthesized = synth.encode(df_encode=pd.concat((df_orig, df_new), axis=0))

    return total_latent_space_usage(df_latent, usage_type='mean')


def latent_dimension_usage(df_latent: pd.DataFrame, usage_type: str = 'mean') -> pd.DataFrame:
    """Calculates the 'usage' of each dimension in the latent space. This is normally a score between 0 and 1.

    A score less than 0.1 would represent an unused latent dimension.

    Args:
        df_latent: The dataset encoded into a latent space as a pandas dataframe.
        usage_type: There are two methods to calculate the usage: 'mean' or 'stddev'.

    Returns:
        The usage in each dimension of the latent space as a pandas dataframe.
    """
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


def total_latent_space_usage(df_latent: pd.DataFrame, usage_type: str = 'mean') -> float:
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
    dims = latent_dimension_usage(df_latent_orig, 'mean')['dimension'][-num_dims:].to_list()[::-1]
    df_latent, df_syn = synth.encode(new_data)

    mean = df_latent.loc[:, [f'm_{d}' for d in dims]].rename(columns=lambda x: int(x[2:]))
    stddev = df_latent.loc[:, [f's_{d}' for d in dims]].rename(columns=lambda x: int(x[2:]))

    orig_mean = df_latent_orig.loc[:, [f'm_{d}' for d in dims]].rename(columns=lambda x: int(x[2:]))
    orig_stddev = df_latent_orig.loc[:, [f's_{d}' for d in dims]].rename(columns=lambda x: int(x[2:]))

    N = len(df_latent)
    N2 = len(df_latent_orig)
    KL = []

    for dim in dims:
        m, v = mean[dim].to_numpy()[:N], stddev[dim].to_numpy()[:N] ** 2
        m_orig, v_orig = orig_mean[dim].to_numpy()[:N2], orig_stddev[dim].to_numpy()[:N2] ** 2

        d = 0.02
        X = np.arange(-5, 5, d)
        Y = np.array([density(x, m, v) / N for x in X])
        Y2 = np.array([density(x, m_orig, v_orig) / N2 for x in X])

        KL.append(d * np.sum(Y * np.log(Y / Y2)))

    return np.sum(KL)


def dataset_quality_by_chunk_old(df: pd.DataFrame, n: int = 10, synth: Optional[HighDimSynthesizer] = None,
                                 progress_callback: Callable[[int], None] = None) -> pd.DataFrame:

    if progress_callback is not None:
        progress_callback(0)

    if synth is None:
        df_meta = MetaExtractor.extract(df)
        synth = HighDimSynthesizer(df_meta=df_meta)

    if progress_callback is not None:
        progress_callback(5)

    size = len(df) // n

    df_cumulative = pd.DataFrame()
    df_results = pd.DataFrame(index=pd.Index(data=[], name='num_rows', dtype=int))

    with synth as synthesizer:
        for i in range(n):
            chunk = df.iloc[i * size:(i + 1) * size]

            synthesizer.learn(df_train=chunk, num_iterations=None)
            df_cumulative = df_cumulative.append(chunk)

            df_latent, df_synthesized = synth.encode(df_encode=df_cumulative)
            ldu = latent_dimension_usage(df_latent, usage_type='mean')

            df_results = df_results.append(
                pd.DataFrame(
                    data={'quality': sum(ldu[ldu['usage'] > 0.1]['usage'])},
                    index=pd.Index(data=[(i + 1) * size], name='num_rows')
                )
            )

            if progress_callback is not None:
                progress_callback((i + 1) * 100 // n)

    return df_results


def dataset_quality_by_chunk(df: pd.DataFrame, n: int = 10, synth: Optional[HighDimSynthesizer] = None,
                             progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
    """Segments a dataframe into chunks and computes the quality score over an increasing number of chunks

    Args:
        df: The dataframe to compute the quality score with
        n: The number of chunks to split the data frame into.
        synth: Optional. A synthesizer that has been trained on a similar data frame.
        progress_callback: A progress callback for the frontend.

    Returns:
        A dataframe with the column "quality" and index "num_rows".
    """

    if progress_callback is not None:
        progress_callback(0)

    if synth is None:
        df_meta = MetaExtractor.extract(df)
        synth = HighDimSynthesizer(df_meta=df_meta)

        with synth as synthesizer:
            synthesizer.learn(df_train=df, num_iterations=None)

    if progress_callback is not None:
        progress_callback(50)

    size = len(df) // n
    df_latent, df_synthesized = synth.encode(df_encode=df)

    df_cumulative = pd.DataFrame()
    df_results = pd.DataFrame(index=pd.Index(data=[], name='num_rows', dtype=int))

    with synth as synthesizer:
        for i in range(n):
            chunk = df.iloc[i * size:(i + 1) * size]
            df_cumulative = df_cumulative.append(chunk)

            quality = latent_kl_difference(
                synth=synthesizer, df_latent_orig=df_latent, new_data=df_cumulative, num_dims=16
            )

            df_results = df_results.append(
                pd.DataFrame(
                    data={'quality': quality},
                    index=pd.Index(data=[(i + 1) * size], name='num_rows')
                )
            )

            if progress_callback is not None:
                progress_callback(50 + (i + 1) * 50 // n)

    return df_results
