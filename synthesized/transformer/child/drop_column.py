from typing import Any, Optional, cast

import pandas as pd

from ..base import Transformer
from ..exceptions import FittingError, NonInvertibleTransformError
from ...metadata_new import Nominal


class DropColumnTransformer(Transformer):
    """
    Drop a column

    Attributes:
        name (str) : the data frame column to transform.
    """

    def __init__(self, name: str):
        super().__init__(name=name)

    def fit(self, df: pd.DataFrame) -> 'DropColumnTransformer':
        return cast(DropColumnTransformer, super().fit(df))

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df.drop(self.name, axis=1, errors='ignore')

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NonInvertibleTransformError


class DropConstantColumnTransformer(Transformer):
    """
    Drop a column

    Attributes:
        name (str) : the data frame column to transform.
    """
    def __init__(self, name: str, constant_value: Optional[Any] = None):
        super().__init__(name=name)
        self._constant_value = constant_value

    def fit(self, df: pd.DataFrame) -> 'DropConstantColumnTransformer':
        if self._constant_value is None:
            counts = pd.value_counts(df[self.name])
            if len(counts) != 1:
                raise FittingError
            self._constant_value = counts.index[0]
        return cast(DropConstantColumnTransformer, super().fit(df))

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df.drop(self.name, axis=1, errors='ignore')

    def inverse_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df[self.name] = self._constant_value
        return df

    @classmethod
    def from_meta(cls, meta: Nominal) -> 'DropConstantColumnTransformer':
        constant_value = meta.categories[0] if meta.categories is not None and len(meta.categories) == 1 else None
        return DropConstantColumnTransformer(name=meta.name, constant_value=constant_value)
