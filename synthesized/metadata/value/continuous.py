from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..base import Ring, Scale


class Integer(Scale[np.int64]):
    dtype = 'i8'
    precision = np.int64(1)

    def __init__(
            self, name: str, categories: Optional[Sequence[np.int64]] = None, nan_freq: Optional[float] = None,
            min: Optional[np.int64] = None, max: Optional[np.int64] = None, num_rows: Optional[int] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows, min=min, max=max)

    def extract(self, df: pd.DataFrame) -> 'Integer':
        sub_df = pd.to_numeric(df[self.name], errors='coerce').to_frame()
        super().extract(sub_df)
        return self


class Float(Ring[np.float64]):
    dtype = 'f8'
    precision = np.float64(0.)

    def __init__(
            self, name: str, categories: Optional[Sequence[np.float64]] = None, nan_freq: Optional[float] = None,
            min: Optional[np.float64] = None, max: Optional[np.float64] = None, num_rows: Optional[int] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows, min=min, max=max)

    def extract(self, df: pd.DataFrame) -> 'Float':
        sub_df = pd.to_numeric(df[self.name], errors='coerce').to_frame()
        super().extract(sub_df)
        return self
