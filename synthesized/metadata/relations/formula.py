import inspect
from typing import Callable

import pandas as pd

from .relation import Relation


class Formula(Relation):
    def __init__(self, name: str, formula: Callable):
        super().__init__(name)
        self.formula = formula

    def get_rules_str(self):
        return f"{self.name}:\n* {inspect.getsourcelines(self.formula)[0][0][:-1]}"

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Here is were validity rules are applied"""
        df = super().postprocess(df)

        df[self.name] = df.apply(self.formula, axis=1)
        return df
