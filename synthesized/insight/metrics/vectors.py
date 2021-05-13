from typing import Optional, Union

import pandas as pd

from .metrics_base import ColumnVector
from ...model import DataFrameModel


class DiffVector(ColumnVector):
    name = 'diff_vector'

    def __init__(self, periods: int = 1):
        self.periods = periods

    def __call__(self, sr: pd.Series, df_model: Optional[DataFrameModel] = None) -> Union[pd.Series, None]:
        return sr.diff(periods=self.periods)


class FractionalDiffVector(ColumnVector):
    name = 'fractional_diff_vector'

    def __init__(self, periods: int = 1):
        self.periods = periods

    def __call__(self, sr: pd.Series, df_model: Optional[DataFrameModel] = None) -> Union[pd.Series, None]:
        return sr.diff(periods=self.periods) / sr
