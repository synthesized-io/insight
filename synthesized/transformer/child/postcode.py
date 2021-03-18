from typing import List, Optional, Pattern, Sequence

import pandas as pd

from ..base import Transformer
from ...model.models import PostcodeModel
from ...util import get_postcode_key_from_df


class PostcodeTransformer(Transformer):
    """
    Transform a address columns into postcodes.

    Attributes:
        name (str) : the data frame column to transform.
        postcode_regex (Pattern[str]): the regex used to extract postcodes.
        postcode_label (Optional[str]): The name of the postcode column.
        full_address_label (Optional[str]): Name of the column containing the full address.
        postcodes (Optional[Sequence[str]]): Optional Series of postcodes to filter to.
    """

    def __init__(self, name: str, postcode_regex: Pattern[str], postcode_label: Optional[str] = None,
                 full_address_label: Optional[str] = None, postcodes: Optional[Sequence[str]] = None):

        super().__init__(name=name)

        self.postcode_regex = postcode_regex
        self.postcode_label = postcode_label
        self.full_address_label = full_address_label
        self.postcodes = postcodes

        self._fitted = True

    def __repr__(self):
        return (f'{self.__class__.__name__}(name="{self.name}", postcode_label="{self.postcode_label}", '
                f'full_address_label="{self.full_address_label}")')

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df[self.name] = get_postcode_key_from_df(
            df, postcode_regex=self.postcode_regex,
            postcode_label=self.postcode_label, full_address_label=self.full_address_label,
            postcodes=self.postcodes)
        df.drop(columns=self.postcode_label, inplace=True)
        return df

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df

    @classmethod
    def from_meta(cls, meta: PostcodeModel) -> 'PostcodeTransformer':
        return cls(meta.name, postcode_regex=meta.postcode_regex, postcode_label=meta.postcode_label,
                   full_address_label=meta.full_address_label, postcodes=meta.categories)

    @property
    def in_columns(self) -> List[str]:
        in_columns = []
        if self.postcode_label is not None:
            in_columns.append(self.postcode_label)
        if self.full_address_label is not None:
            in_columns.append(self.full_address_label)
        return in_columns
