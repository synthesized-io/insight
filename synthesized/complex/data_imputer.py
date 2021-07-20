import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from synthesized.complex import HighDimSynthesizer

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import Synthesizer
from ..common.values import Value

logger = logging.getLogger(__name__)


class DataImputer(Synthesizer):
    """Impute values (e.g missing values, outliers) in original data using a Synthesizer to generate realistic
    data points.

    Args:
        synthesizer (HighDimSynthesizer): Trained Synthesizer instance used to impute data.
    """

    def __init__(self, synthesizer: 'HighDimSynthesizer'):
        super().__init__(name='conditional')
        self.synthesizer = synthesizer
        self._global_step = synthesizer._global_step
        self.logdir = synthesizer.logdir
        self._loss_history = synthesizer._loss_history
        self._writer = synthesizer._writer

    def __repr__(self):
        return f"DataImputer(synthesizer={self.synthesizer})"

    def _get_values(self) -> List[Value]:
        return self.synthesizer._get_values()

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[object, int, dict], bool] = None, callback_freq: int = 0
    ) -> None:
        self.synthesizer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def impute_mask(self, df: pd.DataFrame, mask: pd.DataFrame, produce_nans: bool = False,
                    inplace: bool = False, progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        """Imputes values within a dataframe from a given mask using the underlying synthesizer.

        Args:
            df (pd.DataFrame): The data in which to impute values.
            mask (pd.DataFrame): A boolean mask that contains True for those values to be imputed, and False for the
                values to remain unchanged.
            produce_nans (bool, optional): Whether to produce nans when imputing values for given mask.
                Defaults to False.
            inplace (bool, optional): If True, modifies the given dataframe in place. Defaults to False.

        Returns:
            The DataFrame with masked values imputed.
        """
        if df.size != mask.size:
            raise ValueError(f"Given dataframe and mask must have same size, given df.size={df.size} "
                             f"and mask.size={mask.size}")

        if not inplace:
            df = df.copy()

        rows_to_impute = mask.sum(axis=1) > 0

        # If there's nothing to impute
        if rows_to_impute.sum() == 0:
            if progress_callback is not None:
                progress_callback(100)
            return df

        df_encoded = self.synthesizer._encode_deterministic(df[rows_to_impute], produce_nans=produce_nans)
        if progress_callback is not None:
            # 60% of time is spent on encode_deterministic()
            progress_callback(60)

        n_columns = len(df.columns)
        for i, col in enumerate(df.columns):
            index_to_impute = mask[mask[col]].index
            df.loc[df.index.isin(index_to_impute), col] = df_encoded.loc[df_encoded.index.isin(index_to_impute), col]

            if progress_callback is not None:
                # Remaining 40% time is spent on imputing columns
                progress_callback(60 + (i + 1) * 40 // n_columns)

        return df

    def impute_nans(self, df: pd.DataFrame, inplace: bool = False,
                    progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        """Impute NaN values within a dataframe using the underlying synthesizer.

        Args:
            df (pd.DataFrame): The data in which to impute values.
            inplace (bool, optional): If True, modifies the given dataframe in place. Defaults to False.

        Returns:
            The DataFrame with NaN values imputed.
        """

        if progress_callback is not None:
            progress_callback(0)

        if not inplace:
            df = df.copy()

        nans = df.isna()
        if np.sum(nans.values, axis=None) == 0:
            logger.warning("Given df doesn't contain NaNs. Returning it as it is.")
            return df

        self.impute_mask(df, nans, inplace=True, produce_nans=False, progress_callback=progress_callback)
        return df

    def impute_outliers(self, df: pd.DataFrame, outliers_percentile: float = 0.05, inplace: bool = False,
                        progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        """Impute outlier values in a DataFrame that are determined by a percentile threshold.

        Args:
            df (pd.DataFrame): The data in which to impute values.
            outliers_percentile (float, optional): The percentile threshold for classifying outliers. All values
                outside of these percentiles are considered outliers and will be imputed. Defaults to 0.05.
            inplace (bool, optional): If True, modifies the given dataframe in place. Defaults to False.

        Returns:
            The DataFrame with outliers imputed.
        """

        if progress_callback is not None:
            progress_callback(0)

        if not inplace:
            df = df.copy()

        outliers = pd.DataFrame(np.zeros(df.shape), columns=df.columns, dtype=bool)

        for name in df.columns:
            column = df[name]
            if column.dtype.kind in ('f', 'i'):
                percentiles = [outliers_percentile * 100. / 2, 100 - outliers_percentile * 100. / 2]
                start, end = np.percentile(column, percentiles)
                outliers.loc[(end < column) | (column < start), name] = True
                column[column > end] = end
                column[column < start] = start

        self.impute_mask(df, outliers, inplace=True, progress_callback=progress_callback)
        return df

    def _get_losses(self, data: Dict[str, tf.Tensor] = None) -> tf.Tensor:
        return self.synthesizer._get_losses()

    # alias method for learn
    fit = learn
