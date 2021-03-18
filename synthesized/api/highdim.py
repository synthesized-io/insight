from typing import BinaryIO, Callable, List, Optional, Tuple, Union

import pandas as pd

from .data_frame_meta import DataFrameMeta
from .models import ContinuousModel, DiscreteModel
from .synthesizer import Synthesizer
from ..complex import HighDimSynthesizer as _HighDimSynthesizer
from ..config import HighDimConfig
from ..model import ContinuousModel as _ContinuousModel
from ..model import DiscreteModel as _DiscreteModel


class HighDimSynthesizer(Synthesizer):
    """The main synthesizer implementation.

    Synthesizer which can learn from data to produce basic tabular data with independent rows, that
    is, no temporal or otherwise conditional relation between the rows.
    """
    def __init__(self, df_meta: DataFrameMeta, config: HighDimConfig = HighDimConfig(),
                 type_overrides: List[Union[ContinuousModel, DiscreteModel]] = None):
        """Initialize a new BasicSynthesizer instance.

        Args:
            df_meta: Data sample which summarizes all relevant characteristics,
                so for instance all values a discrete-value column can take.
            config: The configuration for the synthesizer.
            type_overrides: list of modelling assumptions that will override the default behaviour of the Synthesizer
        """
        if df_meta._df_meta is None:
            raise ValueError
        super().__init__()
        _type_overrides: Optional[List[Union[_ContinuousModel, _DiscreteModel]]] = None
        if type_overrides is not None:
            _type_overrides = [model._model for model in type_overrides]

        self._synthesizer = _HighDimSynthesizer(df_meta=df_meta._df_meta, config=config, type_overrides=_type_overrides)

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
            self, num_rows: int, progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.
            progress_callback: Progress bar callback.

        Returns:
            The generated data.

        """
        return self._synthesizer.synthesize(
            num_rows=num_rows, progress_callback=progress_callback
        )

    def encode(
            self, df_encode: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encodes dataset and returns the corresponding latent space and generated data.

        Args:
            df_encode: Input dataset

        Returns:
            (Pandas DataFrame of latent space, Pandas DataFrame of decoded space) corresponding to input data.
        """
        return self._synthesizer.encode(df_encode=df_encode)

    def encode_deterministic(
            self, df_encode: pd.DataFrame,
    ) -> pd.DataFrame:
        """Deterministically encodes a dataset and returns it with imputed nans.

        Args:
            df_encode: Input dataset

        Returns:
            Pandas DataFrame of decoded space corresponding to input data.
        """
        return self._synthesizer.encode_deterministic(df_encode=df_encode)

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
