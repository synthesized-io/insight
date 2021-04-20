from abc import ABC, abstractmethod

import pandas as pd


class GenericRule(ABC):
    """Base class mixin for generic rules that are used for rule-based sampling
    of HighDimSynthesizer.

    Attributes:
        name: the dataframe column name.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self):
        return f"GenericRule(name={self.name})"

    @abstractmethod
    def is_valid(self, df: pd.DataFrame) -> pd.Series:
        ...

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows in a dataframe that are not valid under the rule."""
        return df.loc[self.is_valid(df)]
