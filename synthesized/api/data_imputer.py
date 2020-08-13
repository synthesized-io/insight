from typing import Union, Callable, Optional

import pandas as pd

from .synthesizer import Synthesizer
from ..complex.data_imputer import DataImputer as _DataImputer


class DataImputer(Synthesizer):
    """Imputes synthesized values for nans."""

    def __init__(self, synthesizer: Synthesizer):
        """Data Imputer constructor.

        Args:
            synthesizer: Synthesizer used to impute data. If not given, will create a HighDim from df.

        """
        super().__init__()
        self._data_imputer = _DataImputer(synthesizer=synthesizer._synthesizer)

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[object, int, dict], bool] = None, callback_freq: int = 0
    ) -> None:
        self._data_imputer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def synthesize(self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None,
                   progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        return self._data_imputer.synthesize(
            num_rows=num_rows, conditions=conditions, progress_callback=progress_callback
        )

    def impute_nans(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        return self._data_imputer.impute_nans(df=df, inplace=inplace)

    def impute_outliers(self, df: pd.DataFrame, outliers_percentile: float = 0.05,
                        inplace: bool = False) -> pd.DataFrame:
        return self._data_imputer.impute_outliers(df=df, outliers_percentile=outliers_percentile, inplace=inplace)
