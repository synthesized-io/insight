from typing import Optional, Sequence

from ..base import Nominal


class String(Nominal[str]):
    dtype = 'U'

    def __init__(
            self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)
