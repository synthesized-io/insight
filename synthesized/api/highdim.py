from typing import BinaryIO, Callable, List, Optional, Tuple, Union

import pandas as pd

from .data_frame_meta import DataFrameMeta
from .synthesizer import Synthesizer
from ..complex import HighDimSynthesizer as _HighDimSynthesizer
from ..config import HighDimConfig


class HighDimSynthesizer(Synthesizer):

    def __init__(self, df_meta: DataFrameMeta, conditions: List[str] = None, config: HighDimConfig = HighDimConfig()):
        if df_meta._df_meta is None:
            raise ValueError
        super().__init__()
        self._synthesizer = _HighDimSynthesizer(df_meta=df_meta._df_meta, conditions=conditions, config=config)

    def learn(
            self, df_train: pd.DataFrame, num_iterations: Optional[int],
            callback: Callable[['HighDimSynthesizer', int, dict], bool] = None,
            callback_freq: int = 0
    ) -> None:
        self._synthesizer.learn(
            df_train=df_train, num_iterations=num_iterations, callback=callback,
            callback_freq=callback_freq, low_memory=False
        )

    def synthesize(
            self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:
        return self._synthesizer.synthesize(
            num_rows=num_rows, conditions=conditions, progress_callback=progress_callback
        )

    def encode(
            self, df_encode: pd.DataFrame, conditions: Union[dict, pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._synthesizer.encode(df_encode=df_encode, conditions=conditions)

    def encode_deterministic(
            self, df_encode: pd.DataFrame, conditions: Union[dict, pd.DataFrame] = None
    ) -> pd.DataFrame:
        return self._synthesizer.encode_deterministic(df_encode=df_encode, conditions=conditions)

    def export_model(self, fp: BinaryIO, title: str = None, description: str = None, author: str = None):
        self._synthesizer.export_model(fp=fp, title=title, description=description, author=author)

    @staticmethod
    def import_model(fp: BinaryIO):
        synthesizer = HighDimSynthesizer.__new__(HighDimSynthesizer)
        super(HighDimSynthesizer, synthesizer).__init__()
        synthesizer._synthesizer = _HighDimSynthesizer.import_model(fp=fp)

        return synthesizer
