from typing import Optional, List

import numpy as np
import pandas as pd

from .value_meta import ValueMeta


class SamplingMeta(ValueMeta):

    def __init__(self, name: str, uniform: bool = False, smoothing: float = None,
                 categories: List[str] = None, probabilities: List[float] = None):
        super().__init__(name=name)

        if uniform:
            if smoothing is not None and smoothing != 0.0:
                raise NotImplementedError
            self.smoothing = 0.0
        elif smoothing is None:
            self.smoothing = 1.0
        else:
            self.smoothing = smoothing

        self.categories: Optional[pd.Series] = None

        if categories is not None:
            if probabilities is not None:
                if len(categories) != len(probabilities):
                    raise ValueError("Given 'categories' and 'probs' must have same length.")
                self.categories = pd.Series({category: prob for category, prob in zip(categories, probabilities)})

            else:
                self.categories = pd.Series({category: 1 / len(categories) for category in categories})

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(smoothing=self.smoothing)
        return spec

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)
        if self.categories is None:
            self.categories = df.loc[:, self.name].value_counts(normalize=True, sort=True)
            self.categories **= self.smoothing
            self.categories /= self.categories.sum()

    def learned_input_columns(self) -> List[str]:
        return []

    def learned_output_columns(self) -> List[str]:
        return []

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(labels=self.name, axis=1)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        assert self.categories is not None
        df.loc[:, self.name] = np.random.choice(
            a=self.categories.index, size=len(df), p=self.categories.values
        )
        return df
