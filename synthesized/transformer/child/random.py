from typing import Dict

import re
import string

import numpy as np
import pandas as pd
from ..base import Transformer
from ...metadata import ValueMeta


class RandomTransformer(Transformer):
    """
    Transforms by generating a random string with slight format consistency.

    Examples:
        "49050830L" -> "L3n8o3H2M"

    Attributes:
        name (str) : the data frame column to transform.
        str_length (int) : length of the string to be generated.
    """

    def __init__(self, name: str, str_length: int = 10):
        super().__init__(name=name)
        self.string_patterns = {
            string.ascii_lowercase: False,
            string.ascii_uppercase: False,
            string.digits: False,
        }

        self.str_length = str_length
        self.inverse_mappings: Dict[str, str] = {}

    def __repr__(self):
        return f'{self.__class__.__name__}(name="{self.name}")'

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = df[self.name].astype(str)
        for string_pattern in self.string_patterns:
            regex = re.compile(f"[{string_pattern}]+")
            if any(df[self.name].apply(lambda s, pattern=regex: re.search(pattern, s))):
                self.string_patterns[string_pattern] = True

        if self.str_length is None:
            self.str_length = int(np.ceil(df[self.name].apply(len).mean()))

        return super().fit(df)

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df.loc[:, self.name] = [self.generate_random_string(row[self.name]) for idx, row in df.iterrows()]
        return df

    def generate_random_string(self, val) -> str:
        letters = list(''.join([k for k, v in self.string_patterns.items() if v]))
        random_value = ''.join(np.random.choice(letters) for _ in range(self.str_length))
        self.inverse_mappings[random_value] = val
        return random_value

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df.loc[:, self.name] = df.loc[:, self.name].apply(lambda x: self.inverse_mappings[x])
        return df

    @classmethod
    def from_meta(cls, meta: ValueMeta) -> 'RandomTransformer':
        return cls(meta.name)
