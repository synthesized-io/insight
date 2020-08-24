from typing import Callable, Optional, Union

import pandas as pd


class Synthesizer:
    """The base Synthesizer class."""
    def __init__(self):
        self._synthesizer = None

    def learn(
            self, df_train: pd.DataFrame, num_iterations: Optional[int],
            callback: Callable[[object, int, dict], bool] = None, callback_freq: int = 0
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
        raise NotImplementedError

    def synthesize(
            self, num_rows: int, conditions: Union[dict, pd.DataFrame] = None,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.
            conditions: The condition values for the generated rows.
            progress_callback: A callback that receives current percentage of the progress.

        Returns:
            The generated data.

        """
        raise NotImplementedError
