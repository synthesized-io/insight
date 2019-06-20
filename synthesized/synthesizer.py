import base64
from datetime import datetime
import os
from typing import Callable

import pandas as pd

from synthesized.common.module import Module


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


class Synthesizer(Module):

    @staticmethod
    def logging(iteration, fetched):
        print('\niteration: {}'.format(iteration + 1))
        print(', '.join('{}={:1.2e}'.format(name, value) for name, value in fetched.items()))

    def learn(
        self, num_iterations: int, callback: Callable[[int, dict], None] = logging,
        callback_freq: int = 0, **kwargs
    ) -> None:
        """Train the generative model for the given iterations.

        Repeated calls continue training the model, possibly on different data.

        Args:
            num_iterations: The number of training iterations (not epochs).
            callback: A callback function, e.g. for logging purposes. Aborts training if the return
                value is True.
            callback_freq: Callback frequency.

        """
        raise NotImplementedError

    def synthesize(self, num_rows: int, **kwargs) -> pd.DataFrame:
        """Generate the given number of new data rows.

        Args:
            num_rows: The number of rows to generate.

        Returns:
            The generated data.

        """
        raise NotImplementedError
