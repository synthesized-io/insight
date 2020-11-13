from typing import Optional

import pandas as pd


class ValidityRule:
    def __init__(self, name: str, **kwargs):
        self.name = name

    @classmethod
    def extract(cls, name: str, df: pd.DataFrame) -> Optional['ValidityRule']:
        raise NotImplementedError

    def verify(self, sr: pd.Series) -> pd.Series:
        return sr.apply(self.verify_row)

    def verify_row(self, row) -> bool:
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError
