from typing import Optional, Sequence

import numpy as np

from ..base import Ordinal, Ring


class Bool(Ordinal[np.bool8]):
    dtype = '?'

    def __init__(
            self, name: str, categories: Optional[Sequence[np.bool8]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)


class IntegerBool(Ring[np.int64]):
    dtype = 'i8'
    precision = np.int64(1)

    def __init__(
            self, name: str, categories: Optional[Sequence[np.int64]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)
