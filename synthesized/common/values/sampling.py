import numpy as np
import pandas as pd

from .value import Value


class SamplingValue(Value):

    def __init__(self, name: str, uniform: bool = False, smoothing: float = None):
        super().__init__(name=name)

        if uniform:
            if smoothing is not None and smoothing != 0.0:
                raise NotImplementedError
            self.smoothing = 0.0
        elif smoothing is None:
            self.smoothing = 1.0
        else:
            self.smoothing = smoothing

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(smoothing=self.smoothing)
        return spec

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)
        self.categories = df[self.name].value_counts(normalize=True, sort=True, dropna=False)
        self.categories **= self.smoothing
        self.categories /= self.categories.sum()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(labels=self.name, axis=1)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        df.loc[:, self.name] = np.random.choice(
            a=self.categories.index, size=len(df), p=self.categories.values
        )
        return df

    def learned_input_size(self) -> int:
        return 0

    def learned_output_size(self) -> int:
        return 0