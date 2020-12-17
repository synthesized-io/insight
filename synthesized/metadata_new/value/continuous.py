from typing import Optional, MutableSequence

from ..base import Scale, Ring


class Integer(Scale[int]):
    class_name: str = 'Integer'
    dtype = int
    precision: int = 1

    def __init__(
            self, name: str, categories: Optional[MutableSequence[int]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)


class Float(Ring[float]):
    class_name: str = 'Float'
    dtype = float
    precision: float = 0.

    def __init__(
            self, name: str, categories: Optional[MutableSequence[float]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)
