from typing import List, Union

import numpy as np
import pandas as pd

from .base import GenericRule


class ValueRange(GenericRule):
    """Constrain data to a specific range.

    The range can be defined either in terms of numeric bounds, e.g 0 < x < 10, or with reference to
    another column, e.g x > y.

    Args:
        name (str): the dataframe column name that this rule applies to.
        low (Union[int, float, str]): the lower bound. If a string is given this must refer to another column.
        high (Union[int, float, str]): the upper bound. If a string is given this must refer to another column.
        low_inclusive (bool, optional): whether the range includes the lower bound. Defaults to False.
        high_inclusive (bool, optional): whether range includes the upper bound. Defaults to False.

    Raises:
        ValueError: If 'high' < 'low'.
    """

    def __init__(
        self,
        name: str,
        low: Union[int, float, str] = -np.inf,
        high: Union[int, float, str] = np.inf,
        low_inclusive: bool = False,
        high_inclusive: bool = False,
    ) -> None:
        super().__init__(name)
        self.low = low
        self.high = high
        self.low_inclusive = low_inclusive
        self.high_inclusive = high_inclusive

        if (not isinstance(low, str) and not isinstance(high, str)) and self.low > self.high:  # type: ignore # will only get here if bounds are numeric
            raise ValueError("Upper bound must be greater than lower bound.")

    def __repr__(self):
        return f"ValueRange({self.name}, low={self.low}, high={self.high}, low_inclusive={self.low_inclusive}, high_inclusive={self.high_inclusive})"

    def _is_valid(self, df: pd.DataFrame) -> pd.Series:
        return df.eval(self._to_str())

    def _to_str(self):
        comparator = "<=" if self.low_inclusive else "<"
        s = f"{self.low} {comparator} {self.name}"

        comparator = ">=" if self.high_inclusive else ">"
        s += f" & {self.high} {comparator} {self.name}"

        return s


class ValueEquals(GenericRule):
    """Constrain data to be equal to a specific value.

    Args:
        name (str): the dataframe column name that this rule applies to.
        value (Union[int, float, str]): the value that the rows in this column should take.
    """

    def __init__(self, name: str, value: Union[int, float, str]) -> None:
        super().__init__(name)
        self.value = value

    def __repr__(self):
        return f"ValueEquals(name={self.name}, value={self.value})"

    def _is_valid(self, df: pd.DataFrame) -> pd.Series:
        return df.eval(f"{self.name} == {self.value}")


class ValueIsIn(GenericRule):
    """Constrain data to be within a list of specified values.

    Args:
        values (List[Union[int, float, str]]): the list of values.
    """

    def __init__(self, name: str, values: List[Union[int, float, str]]) -> None:
        super().__init__(name)
        self.values = values

    def __repr__(self):
        return f"ValueIsIn(name={self.name}, values={self.values})"

    def _is_valid(self, df: pd.DataFrame) -> pd.Series:
        return df.eval(f"{self.name} in {self.values}")


class CaseWhenThen(GenericRule):
    """Combine two generic rules, such that when one is valid the second must also be valid, e.g
    when x > 10, then y < 2

    Args:
        when (GenericRule): a generic value rule.
        then (GenericRule): a generic value rule to combine with the when rule.

    Raises:
        ValueError: If 'when.name' == 'then.name'.
    """

    def __init__(self, when: GenericRule, then: GenericRule) -> None:
        if when.name == then.name:
            raise ValueError("Rules must refer to different columns.")

        super().__init__(name="f{when.name};{then.name}")
        self.when = when
        self.then = then

    def __repr__(self):
        return f"CaseWhenThen(when={self.when}, then={self.then})"

    def _is_valid(self, df: pd.DataFrame) -> pd.Series:
        return ~(self.when._is_valid(df) & ~self.then._is_valid(df))
