from typing import List

import numpy as np
import pandas as pd

from ..validity_rules import ValidityRule
from ..values import ValueMeta


class ValidityValue(ValueMeta):
    """Parent class for validity values"""

    def __init__(self, name: str, validity_rules: List[ValidityRule] = None):
        super().__init__(name)

        self.validity_rules: List[ValidityRule] = [] if validity_rules is None else validity_rules

    def extract(self, df: pd.DataFrame) -> None:
        """Extract Validity Rules"""
        pass

    def get_rules_str(self):
        return '\n'.join([f"* {r.name}: {r}" for r in self.validity_rules])

    def learned_input_columns(self) -> List[str]:
        """Validity values do not need a generative model, therefore no column is learned"""
        return []

    def learned_output_columns(self) -> List[str]:
        """Validity values do not need a generative model, therefore no column is learned"""
        return []

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validity values don't need preprocessing."""
        return super().preprocess(df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Here is were validity rules are applied"""
        df = super().preprocess(df)
        num_rows = len(df)
        df.loc[:, self.name] = np.random.rand(num_rows)

        for rule in self.validity_rules:
            df.loc[:, self.name] = rule.transform(df.loc[:, self.name])

        return df

    def add_validity_rule(self, validity_rule: ValidityRule):
        """Add a Validity Rule to this Validity Value"""
        self.validity_rules.append(validity_rule)

    def add_validity_rules(self, validity_rules: List[ValidityRule]):
        """Add a list of Validity Rule to this Validity Value"""
        for validity_rule in validity_rules:
            self.add_validity_rule(validity_rule)
