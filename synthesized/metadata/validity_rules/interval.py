from typing import Union

import pandas as pd

from .validity_rule import ValidityRule


class Interval(ValidityRule):

    def __init__(self, name: str, min_val: Union[int, float] = None, max_val: Union[int, float] = None):
        super().__init__(name)

        self.min_val = min_val
        self.max_val = max_val
        # self.min_included = min_included
        # self.max_included = max_included

    def __str__(self) -> str:
        return f"{self.min_val} <= x < {self.max_val}"

    @classmethod
    def extract(cls, name: str, sr: pd.Series) -> ValidityRule:
        min_val = sr.min()
        max_val = sr.max()

        interval = cls(name, min_val=min_val, max_val=max_val)

        # assert interval.verify(sr).all()
        return interval

    def verify_row(self, x) -> bool:
        return True if self.min_val <= x < self.max_val else False

    def transform(self, sr: pd.Series) -> pd.Series:
        return sr.apply(self.transform_sample)

    def transform_sample(self, x):
        return x * (self.max_val - self.min_val) + self.min_val
