from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..base import Ring, Scale, ValueMeta


class Integer(Scale[np.int64]):
    dtype = 'i8'
    precision = np.int64(1)

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[np.int64]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, unit_meta: Optional['Integer'] = None
    ):
        super().__init__(
            name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows,
            unit_meta=unit_meta
        )

    def extract(self, df: pd.DataFrame) -> 'Integer':
        sub_df = pd.to_numeric(df[self.name], errors='coerce').to_frame()
        super().extract(sub_df)
        return self

    def update_meta(self, df: pd.DataFrame) -> 'Integer':
        sub_df = pd.to_numeric(df[self.name], errors='coerce').to_frame()
        super().update_meta(sub_df)
        return self


class Float(Ring[np.float64]):
    dtype = 'f8'
    precision = np.float64(0.)

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[np.float64]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, unit_meta: Optional['Float'] = None
    ):
        super().__init__(
            name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows,
            unit_meta=unit_meta
        )

    def extract(self, df: pd.DataFrame) -> 'Float':
        sub_df = pd.to_numeric(df[self.name], errors='coerce').to_frame()
        super().extract(sub_df)
        return self

    def update_meta(self, df: pd.DataFrame) -> 'Float':
        sub_df = pd.to_numeric(df[self.name], errors='coerce').to_frame()
        super().update_meta(sub_df)
        return self
