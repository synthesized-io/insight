from typing import Optional, Sequence

import numpy as np

from ..base import Ring, Ordinal


class Bool(Ordinal[np.bool]):
    class_name: str = 'Bool'
    dtype = '?'

    def __init__(
            self, name: str, categories: Optional[Sequence[bool]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)


class IntegerBool(Ring[np.int64]):
    class_name: str = 'IntegerBool'
    dtype = 'i8'
    precision = np.int64(1)

    def __init__(
            self, name: str, categories: Optional[Sequence[np.int64]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)
