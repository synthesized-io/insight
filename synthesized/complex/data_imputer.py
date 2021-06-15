import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common import Synthesizer
from ..common.values import Value

logger = logging.getLogger(__name__)


class DataImputer(Synthesizer):
    """Imputes synthesized values for nans."""

    def __init__(self, synthesizer: Synthesizer):
        """Data Imputer constructor.

        Args:
            synthesizer: Synthesizer used to impute data. If not given, will create a HighDim from df.

        """
        super().__init__(name='conditional')
        self.synthesizer = synthesizer
        self.global_step = synthesizer.global_step
        self.logdir = synthesizer.logdir
        self.loss_history = synthesizer.loss_history
        self.writer = synthesizer.writer

    def get_values(self) -> List[Value]:
        return self.synthesizer.get_values()

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
            df: A pandas DataFrame.
            mask: A boolean pandas DataFrame, containg True for those values to be imputed.
            produce_nans: Whether to produce nans when imputing values for given mask.
            inplace: If true, modifies the given dataframe in place.

        Returns:
            The DataFrame with values imputed.
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

        df_encoded = self.synthesizer.encode_deterministic(df[rows_to_impute], produce_nans=produce_nans)
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
        """Imputes NaN values within a dataframe using the underlying synthesizer.

        Args:
            df: A pandas DataFrame containing NaN values.
            inplace: If true, modifies the given dataframe in place.

        Returns:
            The DataFrame with its NaN values imputed.
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
        """Imputes values in a DataFrame that our outliers by a given threshold.

        Args:
            df: A pandas DataFrame containing NaN values.
            outliers_percentile: The percentile threshold for classifying outliers.
            inplace: If true, modifies the given dataframe in place.

        Returns:
            The DataFrame with its outliers imputed.
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

    def get_losses(self, data: Dict[str, tf.Tensor] = None) -> tf.Tensor:
        return self.synthesizer.get_losses()
