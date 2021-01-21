from typing import Optional

import numpy as np

from .base import Domain, Ring, Scale


class Integer(Scale[np.int64]):
    class_name: str = 'Integer'
    dtype: str = 'int64'

    def __init__(
            self, name: str, domain: Optional[Domain[np.int64]] = None, nan_freq: Optional[float] = None,
            min: Optional[np.int64] = None, max: Optional[np.int64] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)


class Float(Ring[np.float64]):
    class_name: str = 'Float'
    dtype: str = 'float64'

    def __init__(
            self, name: str, domain: Optional[Domain[np.float64]] = None, nan_freq: Optional[float] = None,
            min: Optional[np.float64] = None, max: Optional[np.float64] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)
