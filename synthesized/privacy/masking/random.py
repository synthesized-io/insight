import random
import re
import string

import numpy as np
import pandas as pd

from .base_mask import BaseMask


class RandomMask(BaseMask):
    def __init__(self, column_name: str, str_length: int = 10):
        super(RandomMask, self).__init__(column_name)

        self.string_patterns = {
            string.ascii_lowercase: False,
            string.ascii_uppercase: False,
            string.digits: False,
        }

        self.str_length = str_length

    def fit(self, df: pd.DataFrame):
        df[self.column_name] = df[self.column_name].astype(str)
        for string_pattern in self.string_patterns:
            regex = re.compile(f"[{string_pattern}]+")
            if any(df[self.column_name].apply(lambda s: re.search(regex, s))):
                self.string_patterns[string_pattern] = True

        if self.str_length is None:
            self.str_length = int(np.ceil(df[self.column_name].apply(len).mean()))

        super(RandomMask, self).fit(df)

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        df = super(RandomMask, self).transform(df, inplace)
        df.loc[:, self.column_name] = [self.generate_random_string() for _ in range(len(df))]
        return df

    def generate_random_string(self) -> str:
        letters = ''.join([k for k, v in self.string_patterns.items() if v])
        return ''.join(random.choice(letters) for _ in range(self.str_length))
