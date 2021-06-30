"""This module implements the Synthesizer base class."""
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import tensorflow as tf

from ..common.values import Value


class Synthesizer:
    def __init__(self, name: str, summarizer_dir: str = None, summarizer_name: str = None):
        self._name = name
        self._summarizer_dir = summarizer_dir
        self._summarizer_name = summarizer_name

        self._global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        tf.summary.experimental.set_step(self._global_step)

        self.logdir = None
        self._loss_history: List[dict] = list()

        # Set up logging.
        if summarizer_dir is not None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            if summarizer_name is not None:
                self.logdir = f"{summarizer_dir}/{summarizer_name}_{stamp}"
            else:
                self.logdir = f"{summarizer_dir}/{stamp}"

            self._writer: tf.summary.SummaryWriter = tf.summary.create_file_writer(self.logdir)
        else:
            self._writer = tf.summary.create_noop_writer()

    def _get_values(self) -> List[Value]:
        raise NotImplementedError()

    def _get_all_values(self) -> List[Value]:
        return self._get_values()

    def _get_data_feed_dict(self, df: pd.DataFrame) -> Dict[str, Sequence[tf.Tensor]]:
        raise NotImplementedError

    def _get_losses(self, data: Dict[str, tf.Tensor] = None) -> tf.Tensor:
        raise NotImplementedError

    @staticmethod
    def _logging(synthesizer, iteration, fetched):
        print('\niteration: {}'.format(iteration))
        print(', '.join('{}={:1.2e}'.format(name, value) for name, value in fetched.items()))
        return False

    def learn(
            self, df_train: pd.DataFrame,
            num_iterations: Optional[int],
            callback: Callable[[object, int, dict], bool] = None,
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
        raise NotImplementedError

    def synthesize(
            self, num_rows: int,
            produce_nans: bool = False,
            progress_callback: Callable[[int], None] = None
    ) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.
            produce_nans: Whether to produce NaNs.
            progress_callback: A callback that receives current percentage of the progress.

        Returns:
            The generated data.

        """
        raise NotImplementedError

    def _get_variables(self) -> Dict[str, Any]:
        return dict(
            name=self._name,
            global_step=self._global_step.numpy()
        )

    def _set_variables(self, variables: Dict[str, Any]):
        assert self._name == variables['name']

        self._global_step.assign(variables['global_step'])
        tf.summary.experimental.set_step(self._global_step)
