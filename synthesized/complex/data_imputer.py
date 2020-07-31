import logging
from typing import Optional, Callable, List, Union, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from ..common.values import Value
from ..common import Synthesizer
from ..metadata import ValueMeta
from ..version import versionadded

logger = logging.getLogger(__name__)


@versionadded('1.0.0')
class DataImputer(Synthesizer):
    """Imputes synthesized values for nans."""

    def __init__(self, synthesizer: Synthesizer):
        """Data Imputer constructor.

        Args:
            synthesizer: Synthesizer used to impute data. If not given, will create a HighDim from df.

        """
        assert not synthesizer.value_factory.produce_nans
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

    @versionadded('1.0.0')
    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[object, int, dict], bool] = Synthesizer.logging, callback_freq: int = 0
    ) -> None:
        self.synthesizer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def _impute_mask(self, df: pd.DataFrame, mask: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        if not inplace:
            df = df.copy()

        to_impute_indexes = df[mask.sum(axis=1) > 0].index
        df_encoded = self.synthesizer.encode_deterministic(df.iloc[to_impute_indexes])
        df_encoded = df_encoded.set_index(to_impute_indexes)
        df[mask] = df_encoded[mask]

        return df

    @versionadded('1.0.0')
    def impute_nans(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        if not inplace:
            df = df.copy()

        nans = df.isna()
        if np.sum(nans.values, axis=None) == 0:
            logger.warning("Given df doesn't contain NaNs. Returning it as it is.")
            return df

        self._impute_mask(df, nans, inplace=True)
        return df

    @versionadded('1.0.0')
    def impute_outliers(self, df: pd.DataFrame, outliers_percentile: float = 0.05,
                        inplace: bool = False) -> pd.DataFrame:
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

        self._impute_mask(df, outliers, inplace=True)
        return df

    @versionadded('1.0.0')
    def synthesize(self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None,
                   progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        return self.synthesizer.synthesize(num_rows=num_rows, conditions=conditions)

    def get_conditions(self) -> List[Value]:
        return self.synthesizer.get_conditions()

    def get_value_meta_pairs(self) -> List[Tuple[Value, ValueMeta]]:
        return self.synthesizer.get_value_meta_pairs()

    def get_condition_meta_pairs(self) -> List[Tuple[Value, ValueMeta]]:
        return self.synthesizer.get_condition_meta_pairs()

    def get_losses(self, data: Dict[str, tf.Tensor] = None) -> tf.Tensor:
        return self.synthesizer.get_losses()
