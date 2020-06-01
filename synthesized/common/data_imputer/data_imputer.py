from typing import Optional, Callable, List, Union, Any, Dict

import pandas as pd

from ...highdim import HighDimSynthesizer
from ..values import Value, CategoricalValue, NanValue
from ..synthesizer import Synthesizer


class DataImputer(Synthesizer):
    """Imputes synthesized values for nans."""

    def __init__(self,
                 synthesizer: Synthesizer = None,
                 highdim_kwargs: Dict[str, Any] = None,
                 df: pd.DataFrame = None):
        """Data Imputer constructor.

        Args:
            synthesizer: Synthesizer used to impute data. If not given, will create a HighDim from df.
            highdim_kwargs: Dictionary containing any extra kwargs for HighDim.
            df: Original DataFrame, not needed if synthesizer is given.

        """

        assert (synthesizer is not None) or (df is not None)
        assert not (synthesizer is not None and df is not None)

        if synthesizer is not None:
            assert not synthesizer.produce_nans_for
            self.synthesizer = synthesizer
        else:
            highdim_kwargs = highdim_kwargs if highdim_kwargs is not None else dict()
            self.synthesizer = HighDimSynthesizer(
                df=df,
                produce_nans_for=None,
                **highdim_kwargs
            )

        self.nan_columns = self.get_nan_columns()

    def __enter__(self):
        return self.synthesizer.__enter__()

    def get_values(self) -> List[Value]:
        return self.synthesizer.get_values()

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[object, int, dict], bool] = Synthesizer.logging, callback_freq: int = 0
    ) -> None:
        self.synthesizer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def impute_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df_synthesized = self.synthesizer.encode_deterministic(df)

        for c in self.nan_columns:
            nans = df.loc[:, c].isna()
            df.loc[nans, c] = df_synthesized.loc[nans, c]

        return df

    def synthesize(self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None,
                   progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        return self.synthesizer.synthesize(num_rows=num_rows, conditions=conditions)

    def get_nan_columns(self) -> List[str]:
        nan_columns = []
        for value in self.get_values():
            if isinstance(value, CategoricalValue) and value.nans_valid:
                nan_columns.append(value.name)
            elif isinstance(value, NanValue):
                nan_columns.append(value.name)

        return nan_columns
