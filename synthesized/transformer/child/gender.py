from typing import List, Optional

import numpy as np
import pandas as pd

from ..base import Transformer
from ...config import GenderTransformerConfig
from ...metadata_new import Affine


class GenderTransformer(Transformer):
    """
    Transform a continuous distribution using the quantile transform.

    Attributes:
        name (str) : the data frame column to transform.
    """

    def __init__(self, name: str, gender_label: Optional[str] = None, title_label: Optional[str] = None,
                 config: GenderTransformerConfig = GenderTransformerConfig()):

        super().__init__(name=name)

        self.gender_label = gender_label
        self.title_label = title_label

        self.title_mapping = config.title_mapping
        self.gender_mapping = config.gender_mapping
        self.genders = list(self.title_mapping.keys())

        self._fitted = True

    def __repr__(self):
        return (f'{self.__class__.__name__}(name="{self.name}", gender_label="{self.gender_label}", '
                f'title_label="{self.title_label}")')

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.gender_label is not None:
            df[self.name] = df[self.gender_label].astype(str).apply(self.get_gender_from_gender)
        elif self.title_label is not None:
            df[self.name] = df[self.title_label].astype(str).apply(self.get_gender_from_title)
        else:
            raise ValueError("Can't extract gender series as 'gender_label' nor 'title_label' are given.")

        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:

        print("before gender inv transf", df.columns)

        if self.gender_label is not None:
            df[self.gender_label] = df[self.name]
        if self.title_label is not None:
            df[self.title_label] = df[self.name].astype(dtype=str).apply(self.get_title_from_gender)

        print("after gender inv transf", df.columns)

        return df

    def get_gender_from_gender(self, gender: str) -> str:
        gender = gender.strip().upper()
        for k, v in self.gender_mapping.items():
            if gender in v:
                return k
        return np.nan

    def get_gender_from_title(self, title: str) -> str:
        title = title.replace('.', '').strip().upper()
        for k, v in self.title_mapping.items():
            if title in v:
                return k
        return np.nan

    def get_title_from_gender(self, gender: str) -> str:
        gender = self.get_gender_from_gender(gender)
        return self.title_mapping[gender][0] if gender in self.title_mapping.keys() else np.nan

    @classmethod
    def from_meta(cls, meta: Affine,
                  config: GenderTransformerConfig = GenderTransformerConfig()) -> 'GenderTransformer':
        return cls(meta.name, config=config)

    @property
    def in_columns(self) -> List[str]:
        in_columns = []
        if self.gender_label is not None:
            in_columns.append(self.gender_label)
        if self.title_label is not None:
            in_columns.append(self.title_label)
        return in_columns
