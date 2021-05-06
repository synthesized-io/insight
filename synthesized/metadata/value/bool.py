from typing import Optional, Sequence

import numpy as np

from ..base import Ordinal, Ring, ValueMeta


class Bool(Ordinal[np.bool8]):
    dtype = '?'

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[np.bool8]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None
    ):
        super().__init__(name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)


class IntegerBool(Ring[np.int64]):
    dtype = 'i8'
    precision = np.int64(1)

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[np.int64]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, unit_meta: Optional['IntegerBool'] = None
    ):
        super().__init__(
            name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows,
            unit_meta=unit_meta
        )
