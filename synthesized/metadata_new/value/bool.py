from typing import Optional, MutableSequence

import numpy as np

from ..base import Ring


class Bool(Ring[np.bool]):
    class_name: str = 'Bool'
    dtype = bool

    def __init__(
            self, name: str, categories: Optional[MutableSequence[bool]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)
