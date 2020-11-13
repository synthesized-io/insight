from typing import Optional

import numpy as np
import pandas as pd

from .validity_rule import ValidityRule


class ContainsNans(ValidityRule):

    def __init__(self, name: str, prop_nans: float = 0.2):
        super().__init__(name)
        self.prop_nans = prop_nans

    def __str__(self):
        return f"X contains {self.prop_nans * 100}% of NaNs"

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
        sr.iloc[np.random.randint(len(sr))] = np.nan
        return sr
