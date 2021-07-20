from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import pandas as pd


class GenericRule(ABC):
    """Base class mixin for generic rules"""
    def __init__(self) -> None:
        """Initialize the rule"""
        pass

    def __repr__(self):
        return "GenericRule()"

    def _is_valid(self, df: pd.DataFrame) -> pd.Series:
        """Returns a boolean series, if the function is valid."""
        pd_str = self.to_pd_str(df_name="df")
        return eval(pd_str, {"df": df, "pd": pd})

    @abstractmethod
    def get_children(self) -> List["GenericRule"]:
        ...

    @abstractmethod
    def to_pd_str(self, df_name: str) -> str:
        """Get the pandas statement to be evaluated for this rule"""
        ...

    @abstractmethod
    def to_sql_str(self) -> str:
        """Get the SQL statement to be evaluated for this rule"""
        ...

    def __eq__(self, o: object) -> bool:
        if type(o) != type(self):
            return False

        for k, v in self.__dict__.items():
            if k not in o.__dict__:
                return False
            if v != o.__dict__[k]:
                return False

        return True

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows in a dataframe that are not valid under the rule.

        Args:
            df (pd.DataFrame): Data to filter.

        Returns:
            DataFrame with rows removed that are not valid under the rule.
        """
        return df.loc[self._is_valid(df)]

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        """Return the functions needed to augment the dataframe so that the rule is covered.

        The returned object is a Dict[str, Callable], where keys are the columns where the result of the function in
        the values will be applied.

        For example, if get_augment_func returns ``{col_name: func}``, it will be applied as
        >>> df[col_name] = df.apply(func, axis=1)

        """
        return dict()
