from typing import Optional

import pandas as pd

from .validity_rule import ValidityRule


class IsInteger(ValidityRule):

    def __init__(self, name: str):
        super().__init__(name)

    def __str__(self):
        return "isInteger(x)"

    @classmethod
    def extract(cls, name: str, sr: pd.Series) -> Optional[ValidityRule]:
        is_integer = cls(name)

        if is_integer.verify(sr).all():
            return is_integer
        else:
            return None

    def verify_row(self, x) -> bool:
        return True if isinstance(x, int) or x % 1 == 0 else False

    def transform(self, sr: pd.Series) -> pd.Series:
        return sr.apply(self.transform_row)

    def transform_row(self, x):
        return int(x)
