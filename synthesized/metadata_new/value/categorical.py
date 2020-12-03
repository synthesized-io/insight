from typing import Optional

from ..base import Domain, Nominal


class String(Nominal[str]):
    class_name: str = 'String'
    dtype: str = 'str'

    def __init__(
            self, name: str, domain: Optional[Domain[str]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq)
