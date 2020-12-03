from typing import Optional

import numpy as np

from ..base import Domain, Ring


class Bool(Ring[np.bool]):
    class_name: str = 'Bool'
    dtype: str = 'bool'

    def __init__(
            self, name: str, domain: Optional[Domain[np.bool]] = None, nan_freq: Optional[float] = None,
            min: Optional[np.bool] = None, max: Optional[np.bool] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq, min=min, max=max)
