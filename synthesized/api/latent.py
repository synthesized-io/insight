from typing import Union

import pandas as pd

from .highdim import HighDimSynthesizer
from ..insight.latent import get_latent_space as _get_latent_space, get_data_quality as _get_data_quality
from ..insight.latent import latent_dimension_usage as _latent_dimension_usage
from ..insight.latent import total_latent_space_usage as _total_latent_space_usage


def get_latent_space(df: pd.DataFrame, num_iterations=5_000, **kwargs) -> pd.DataFrame:

    return _get_latent_space(df=df, num_iterations=num_iterations, **kwargs)


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
    synth = synthesizer._synthesizer
    return _get_data_quality(synthesizer=synth, df_orig=df_orig, df_new=df_new, **kwargs)


def latent_dimension_usage(df_latent: pd.DataFrame, usage_type: str = 'stddev') -> pd.DataFrame:

    return _latent_dimension_usage(df_latent=df_latent, usage_type=usage_type)


def total_latent_space_usage(df_latent: pd.DataFrame, usage_type: str = 'stddev') -> float:

    return _total_latent_space_usage(df_latent=df_latent, usage_type=usage_type)


__all__ = ['get_latent_space', 'latent_dimension_usage', 'total_latent_space_usage']
