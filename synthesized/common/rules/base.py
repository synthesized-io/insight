from abc import ABC, abstractmethod

import pandas as pd


class GenericRule(ABC):
    """Base class for generic rules that are used for rule-based sampling
    of HighDimSynthesizer.

    Args:
        name (str): the dataframe column name that this rule applies to.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self):
        return f"GenericRule(name={self.name})"

    @abstractmethod
    def _is_valid(self, df: pd.DataFrame) -> pd.Series:
        ...

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows in a dataframe that are not valid under the rule.

        Args:
            df (pd.DataFrame): Data to filter.

        Returns:
            DataFrame with rows removed that are not valid under the rule.
        """
        return df.loc[self._is_valid(df)]
