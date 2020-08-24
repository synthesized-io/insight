from typing import Union, Callable, Optional

import pandas as pd

from .synthesizer import Synthesizer
from ..complex.data_imputer import DataImputer as _DataImputer


class DataImputer(Synthesizer):
    """Imputes synthesized values for nans."""

    def __init__(self, synthesizer: Synthesizer):
        """Data Imputer constructor.

        Args:
            synthesizer: Synthesizer used to impute data. If not given, will create a HighDim from df.

        """
        super().__init__()
        self._data_imputer = _DataImputer(synthesizer=synthesizer._synthesizer)

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[object, int, dict], bool] = None, callback_freq: int = 0
    ) -> None:
        """Trains the underlying generative model for the given iterations.

        Repeated calls continue training the model, possibly on different data.

        Args:
            df_train: The training data.
            num_iterations: The number of training iterations (not epochs).
            callback: A callback function, e.g. for logging purposes. Takes the synthesizer
                instance, the iteration number, and a dictionary of values (usually the losses) as
                arguments. Aborts training if the return value is True.
            callback_freq: Callback frequency.

        """
        self._data_imputer.learn(
            num_iterations=num_iterations, df_train=df_train, callback=callback, callback_freq=callback_freq
        )

    def synthesize(self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None,
                   progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        """Generate the given number of new data rows using the underlying synthesizer.

        Args:
            num_rows: The number of rows to generate.
            conditions: The condition values for the generated rows.
            progress_callback: Progress bar callback.

        Returns:
            The generated data.

        """
        return self._data_imputer.synthesize(
            num_rows=num_rows, conditions=conditions, progress_callback=progress_callback
        )

    def impute_nans(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """Imputes NaN values within a dataframe using the underlying synthesizer.

        Args:
            df: A pandas DataFrame containing NaN values.
            inplace: If true, modifies the given dataframe in place.

        Returns:
            The DataFrame with its NaN values imputed.
        """
        return self._data_imputer.impute_nans(df=df, inplace=inplace)

    def impute_outliers(self, df: pd.DataFrame, outliers_percentile: float = 0.05,
                        inplace: bool = False) -> pd.DataFrame:
        """Imputes values in a DataFrame that our outliers by a given threshold.

        Args:
            df: A pandas DataFrame containing NaN values.
            outliers_percentile: The percentile threshold for classifying outliers.
            inplace: If true, modifies the given dataframe in place.

        Returns:
            The DataFrame with its outliers imputed.
        """
        return self._data_imputer.impute_outliers(df=df, outliers_percentile=outliers_percentile, inplace=inplace)
