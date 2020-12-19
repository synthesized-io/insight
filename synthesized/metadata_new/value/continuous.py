from typing import Optional, MutableSequence

import numpy as np

from ..base import Scale, Ring


class Integer(Scale[np.int64]):
    dtype = 'i8'
    precision = np.int64(1)

    def __init__(
            self, name: str, categories: Optional[MutableSequence[int]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)


class Float(Ring[np.float64]):
    dtype = 'f8'
    precision = np.float64(0.)

    def __init__(
            self, name: str, categories: Optional[MutableSequence[float]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)
