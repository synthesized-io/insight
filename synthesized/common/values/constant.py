import numpy as np
import pandas as pd

from .value import Value


class ConstantValue(Value):

    def __init__(self, name: str, constant_value=None):
        super().__init__(name=name)
        self.constant_value = constant_value

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(constant_value=self.constant_value)
        return spec

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)
        unique_values = df[self.name].unique()
        assert len(unique_values) == 1
        self.constant_value = unique_values[0]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(labels=self.name, axis=1)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        df[self.name] = self.constant_value
        return df

    def learned_input_size(self) -> int:
        return 0

    def learned_output_size(self) -> int:
        return 0
