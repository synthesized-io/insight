import pandas as pd


class Expression:
    """Define a column through a mathematical expression, e.g A = 2 * B + C.

    Args:
        name (str): the dataframe column name that this rule applies to.
        expr (str): a string expression that can be evaluated by numexpr.
    """

    def __init__(self, name: str, expr: str) -> None:
        self.name = name
        self.expr = expr

    def __repr__(self):
        return f"Expression(name={self.name}, expr={self.expr})"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the expression to the dataframe.

        Args:
            df (pd.DataFrame): Data to apply expression.

        Returns:
            DataFrame with expression applied.
        """
        return df.eval(f"{self.name} = {self.expr}")
