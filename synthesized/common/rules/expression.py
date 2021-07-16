from typing import List

import pandas as pd

from .node import Column, GenericRule


class Expression:
    """Define a column through a mathematical expression, e.g A = 2 * B + C.

    Args:
        name (str): the dataframe column name that this rule applies to.
        expr (str): a string expression that can be evaluated by numexpr.
    """

    def __init__(self, column: Column, expr: str) -> None:
        self.column = column
        self.expr = expr

    def __repr__(self):
        return f"Expression(column={self.column}, expr={self.expr})"

    def get_children(self) -> List[GenericRule]:
        return [self.column]

    def to_pd_str(self, df_name: str) -> str:
        return f"{df_name}.eval('{self.column.column_name} = {self.expr}')"

    def to_sql_str(self) -> str:
        raise NotImplementedError

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the expression to the dataframe.

        Args:
            df (pd.DataFrame): Data to apply expression.

        Returns:
            DataFrame with expression applied.
        """
        return df.eval(f"{self.column.column_name} = {self.expr}")
