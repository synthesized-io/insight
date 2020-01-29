"""This module implements the Synthesizer base class."""
import base64
from datetime import datetime
import os
from typing import Callable, Union, List, Optional

import pandas as pd
import tensorflow as tf

from .values import Value


def _check_license():
    try:
        key_env = 'SYNTHESIZED_KEY'
        key_path = '~/.synthesized/key'
        key_path = os.path.expanduser(key_path)
        print("Copyright (C) Synthesized Ltd. - All Rights Reserved")
        license_key = os.environ.get(key_env, None)
        if license_key is None and os.path.isfile(key_path):
            with open(key_path, 'r') as f:
                license_key = f.readlines()[0].rstrip()
        if license_key is None:
            print('No license key detected (env variable {} or {})'.format(key_env, key_path))
            return False
        else:
            print('License key: ' + license_key)
        license_key_bytes = base64.b16decode(license_key.replace('-', ''))
        key = 13
        n = 247
        plain = ''.join([chr((char ** key) % n) for char in list(license_key_bytes)])
        date = datetime.strptime(plain.split(' ')[1], "%Y-%m-%d")
        now = datetime.now()
        if now < date:
            print('Expires at: ' + str(date))
            return True
        if now >= date:
            print('License has been expired')
            return False
        else:
            print('Expires at: ' + str(date))
            return True
    except Exception as e:
        print(e)
        return False


if not _check_license():
    raise Exception('Failed to load license key')


class Synthesizer(tf.Module):
    def __init__(self, name: str, summarizer_dir: str = None, summarizer_name: str = None):
        super(Synthesizer, self).__init__(name=name)

        self.global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        tf.summary.experimental.set_step(self.global_step)

        self.logdir = None
        self.loss_history: List[dict] = list()
        self.writer: Optional[tf.summary.SummaryWriter] = None

        # Set up logging.
        if summarizer_dir is not None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            if summarizer_name is not None:
                self.logdir = f"{summarizer_dir}/{summarizer_name}_{stamp}"
            else:
                self.logdir = f"{summarizer_dir}/{stamp}"

            self.writer = tf.summary.create_file_writer(self.logdir)

    def get_values(self) -> List[Value]:
        raise NotImplementedError()

    def get_conditions(self) -> List[Value]:
        raise NotImplementedError()

    def get_losses(self, data) -> tf.Tensor:
        raise NotImplementedError()

    def preprocess(self, df):
        return df

    @staticmethod
    def logging(synthesizer, iteration, fetched):
        print('\niteration: {}'.format(iteration))
        print(', '.join('{}={:1.2e}'.format(name, value) for name, value in fetched.items()))
        return False

    def learn(
        self, df_train: pd.DataFrame, num_iterations: Optional[int],
        callback: Callable[[object, int, dict], bool] = logging, callback_freq: int = 0
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

    def synthesize(self, num_rows: int,
                   conditions: Union[dict, pd.DataFrame] = None,
                   progress_callback: Callable[[int], None] = None) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.
            conditions: The condition values for the generated rows.
            progress_callback: A callback that receives current percentage of the progress.
        Returns:
            The generated data.

        """
        raise NotImplementedError

    def __enter__(self):
        if self.writer is not None:
            self.writer.set_as_default()
        return self

    def __exit__(self, type, value, traceback):
        if self.writer is not None:
            self.writer.close()
