import pandas as pd


class Expression:
    """Define a column through a mathematical expression, e.g A = 2 * B + C.

    Attributes:
        name: the dataframe column name.
        expr: a string expression that can be evaluated by numexpr.
    """

    def __init__(self, name: str, expr: str) -> None:
        self.name = name
        self.expr = expr

    def __repr__(self):
        return f"Expression(name={self.name}, expr={self.expr})"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the expression rule to the dataframe."""
        return df.eval(f"{self.name} = {self.expr}")
