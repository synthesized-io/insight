import logging
from typing import Optional, Callable, List

import pandas as pd

from ..values import CategoricalValue, NanValue
from ..synthesizer import Synthesizer
from ...highdim import HighDimSynthesizer


class DataImputer(Synthesizer):

    def __init__(self, df: pd.DataFrame):
        self.highdim = HighDimSynthesizer(df=df, produce_nans_for=False, beta=0.)
        self.nan_columns = self.get_nan_columns()

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[object, int, dict], bool] = logging, callback_freq: int = 0
    ) -> None:
        self.synthesizer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def impute_nans(self, df):
        df = df.copy()
        _, df_synthesized = self.highdim.encode(df)

        for c in self.nan_columns:
            nans = df.loc[:, c].isna()
            df.loc[nans, c] = df_synthesized.loc[nans, c]

        return df

    def get_nan_columns(self) -> List[str]:
        nan_columns = []
        for value in self.get_values():
            if isinstance(value, CategoricalValue) and value.nans_valid:
                nan_columns.append(value.name)
            elif isinstance(value, NanValue):
                nan_columns.append(value.name)

        return nan_columns

