from typing import Union

import pandas as pd

from .metrics_base import ColumnVector


class DiffVector(ColumnVector):
    name = 'diff_vector'

    def __call__(self, sr, periods=1, **kwargs) -> Union[pd.Series, None]:
        return sr.diff(periods=periods)


class FractionalDiffVector(ColumnVector):
    name = 'fractional_diff_vector'

    def __call__(self, sr, periods=1, **kwargs) -> Union[pd.Series, None]:
        return sr.diff(periods=periods,)/sr
