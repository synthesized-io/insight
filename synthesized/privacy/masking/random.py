import re
import string
from typing import Dict

import numpy as np
import pandas as pd

from synthesized.metadata import ValueMeta
from synthesized.transformer.base import Transformer


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
        """Fits the given dataframe to the transformer

        Args:
            df: Dataset to fit

        Returns:
            self
        """
        df[self.name] = df[self.name].astype(str)
        for string_pattern in self.string_patterns:
            regex = re.compile(f"[{string_pattern}]+")
            if any(df[self.name].apply(lambda s, pattern=regex: re.search(pattern, s))):
                self.string_patterns[string_pattern] = True

        if self.str_length is None:
            self.str_length = int(np.ceil(df[self.name].apply(len).mean()))

        return super().fit(df)

    def _generate_random_string(self, val) -> str:
        letters = list(''.join([k for k, v in self.string_patterns.items() if v]))
        random_value = ''.join(np.random.choice(letters) for _ in range(self.str_length))
        self.inverse_mappings[random_value] = val
        return random_value

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transforms the given dataframe using fitted transformer

        Args:
            df: Dataset to transform

        Returns:
            Transformed dataset
        """
        df.loc[:, self.name] = [self._generate_random_string(row[self.name]) for idx, row in df.iterrows()]
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Inverse transforms the transformed dataframe to give the riginal dataset

        Args:
            df: Transformed dataset

        Returns:
            Original dataset
        """
        df.loc[:, self.name] = df.loc[:, self.name].apply(lambda x: self.inverse_mappings[x])
        return df

    @classmethod
    def from_meta(cls, meta: ValueMeta) -> 'RandomTransformer':
        return cls(meta.name)
