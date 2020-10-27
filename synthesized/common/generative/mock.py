"""Mock Synthesizer"""
from synthesized.common.synthesizer import Synthesizer
from synthesized.metadata import DataFrameMeta
from typing import Optional, Union, Callable
import pandas as pd
import numpy as np


class MockSynthesizer(Synthesizer):
    """
    Mock synthesizer for testing purposes.

    This does not learn but synthesizes data by resampling
    from the given DataFrame.
    """

    def __init__(self, df_meta: DataFrameMeta, **kwargs):
        super().__init__(name='mock_synthesizer', **kwargs)
        self.df_meta = df_meta
        self._df = pd.DataFrame({name: np.ones(1) for name in df_meta.columns})

    def preprocess(self, df: pd.DataFrame, max_workers: Optional[int] = 4) -> pd.DataFrame:
        return self.df_meta.preprocess(df, max_workers=max_workers)

    def postprocess(self, df: pd.DataFrame, max_workers: Optional[int] = 4) -> pd.DataFrame:
        return self.df_meta.postprocess(df, max_workers=max_workers)

    def learn(
            self, df_train: pd.DataFrame,
            num_iterations: Optional[int],
            callback: Callable[[object, int, dict], bool] = None,
            callback_freq: int = 0
    ) -> None:
        self._df = self.preprocess(df_train)

    def synthesize(
            self, num_rows: int,
            conditions: Union[dict, pd.DataFrame] = None,
            produce_nans: bool = False,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:

        if num_rows <= 0:
            raise ValueError(f"num_rows must be greater than zero, not {num_rows}.")

        samples = self._df.sample(num_rows, replace=True)
        df_synthetic = self.postprocess(samples)

        if not produce_nans:
            df_synthetic = df_synthetic.fillna(method='ffill').fillna(method='bfill')

        return df_synthetic
