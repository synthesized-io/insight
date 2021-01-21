from typing import BinaryIO, Callable, List, Optional, Tuple, Union

import pandas as pd

from .data_frame_meta import DataFrameMeta
from .synthesizer import Synthesizer
from ..complex import HighDimSynthesizer as _HighDimSynthesizer
from ..config import HighDimConfig


class HighDimSynthesizer(Synthesizer):
    """The main synthesizer implementation.

    Synthesizer which can learn from data to produce basic tabular data with independent rows, that
    is, no temporal or otherwise conditional relation between the rows.
    """
    def __init__(self, df_meta: DataFrameMeta, conditions: List[str] = None, config: HighDimConfig = HighDimConfig()):
        """Initialize a new BasicSynthesizer instance.

        Args:
            df_meta: Data sample which summarizes all relevant characteristics,
                so for instance all values a discrete-value column can take.
            conditions: List of column names that serve as conditional values.
            config: The configuration for the synthesizer.
        """
        if df_meta._df_meta is None:
            raise ValueError
        super().__init__()
        self._synthesizer = _HighDimSynthesizer(df_meta=df_meta._df_meta, conditions=conditions, config=config)

    def learn(
            self, df_train: pd.DataFrame, num_iterations: Optional[int],
            callback: Callable[['HighDimSynthesizer', int, dict], bool] = None,
            callback_freq: int = 0
    ) -> None:
        """Train the generative model for the given iterations.

        Repeated calls continue training the model, possibly on different data.

        Args:
            df_train: The training data.
            num_iterations: The number of training iterations (not epochs).
            callback: A callback function, e.g. for logging purposes. Takes the synthesizer
                instance, the iteration number, and a dictionary of values (usually the losses) as
                arguments. Aborts training if the return value is True.
            callback_freq: Callback frequency.

        """
        self._synthesizer.learn(
            df_train=df_train, num_iterations=num_iterations, callback=callback,
            callback_freq=callback_freq, low_memory=False
        )

    def synthesize(
            self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.
            conditions: The condition values for the generated rows.
            progress_callback: Progress bar callback.

        Returns:
            The generated data.

        """
        return self._synthesizer.synthesize(
            num_rows=num_rows, conditions=conditions, progress_callback=progress_callback
        )

    def encode(
            self, df_encode: pd.DataFrame, conditions: Union[dict, pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encodes dataset and returns the corresponding latent space and generated data.

        Args:
            df_encode: Input dataset
            conditions: The condition values for the generated rows.

        Returns:
            (Pandas DataFrame of latent space, Pandas DataFrame of decoded space) corresponding to input data.
        """
        return self._synthesizer.encode(df_encode=df_encode, conditions=conditions)

    def encode_deterministic(
            self, df_encode: pd.DataFrame, conditions: Union[dict, pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Deterministically encodes a dataset and returns it with imputed nans.

        Args:
            df_encode: Input dataset
            conditions: The condition values for the generated rows.

        Returns:
            Pandas DataFrame of decoded space corresponding to input data.
        """
        return self._synthesizer.encode_deterministic(df_encode=df_encode, conditions=conditions)

    def export_model(self, fp: BinaryIO, title: str = None, description: str = None, author: str = None):
        """Exports the synthesizer as a binary in its trained state to the provided file.

        Args:
            fp: The binary file to write to.
            title: An optional title for the model.
            description: An optional description of the model
            author: An optional author.

        """
        self._synthesizer.export_model(fp=fp, title=title, description=description, author=author)

    @staticmethod
    def import_model(fp: BinaryIO):
        """Imports a HighDimSynthesizer object and state from a binary file.

        Args:
            fp: The binary file to load from.

        Returns:
            HighDimSynthesizer object from its saved state.

        """
        synthesizer = HighDimSynthesizer.__new__(HighDimSynthesizer)
        super(HighDimSynthesizer, synthesizer).__init__()
        synthesizer._synthesizer = _HighDimSynthesizer.import_model(fp=fp)

        return synthesizer
