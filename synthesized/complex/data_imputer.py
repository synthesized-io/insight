import logging
from typing import Optional, Callable, List, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common.values import Value
from ..common import Synthesizer
from ..metadata import ValueMeta


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

    def __enter__(self):
        return self.synthesizer.__enter__()

    def get_values(self) -> List[Value]:
        return self.synthesizer.get_values()

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[object, int, dict], bool] = None, callback_freq: int = 0
    ) -> None:
        self.synthesizer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def _impute_mask(self, df: pd.DataFrame, mask: pd.DataFrame, produce_nans: bool = False,
                     inplace: bool = False, progress_callback: Callable[[int], None] = None) -> pd.DataFrame:

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

        if progress_callback is not None:
            progress_callback(0)

        if not inplace:
            df = df.copy()

        nans = df.isna()
        if np.sum(nans.values, axis=None) == 0:
            logger.warning("Given df doesn't contain NaNs. Returning it as it is.")
            return df

        self._impute_mask(df, nans, inplace=True, produce_nans=False, progress_callback=progress_callback)
        return df

    def impute_outliers(self, df: pd.DataFrame, outliers_percentile: float = 0.05, inplace: bool = False,
                        progress_callback: Callable[[int], None] = None) -> pd.DataFrame:

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

        self._impute_mask(df, outliers, inplace=True, progress_callback=progress_callback)
        return df

    def get_conditions(self) -> List[Value]:
        return self.synthesizer.get_conditions()

    def get_value_meta_pairs(self) -> List[Tuple[Value, ValueMeta]]:
        return self.synthesizer.get_value_meta_pairs()

    def get_condition_meta_pairs(self) -> List[Tuple[Value, ValueMeta]]:
        return self.synthesizer.get_condition_meta_pairs()

    def get_losses(self, data: Dict[str, tf.Tensor] = None) -> tf.Tensor:
        return self.synthesizer.get_losses()
