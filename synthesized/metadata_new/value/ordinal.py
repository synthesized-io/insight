from typing import Optional, List, cast, MutableSequence, Dict

import pandas as pd

from ..base import Ordinal
from ..exceptions import MetaNotExtractedError, ExtractionError


class OrderedString(Ordinal[str]):
    class_name: str = 'String'
    dtype: str = 'str'

    def __init__(
            self, name: str, categories: Optional[MutableSequence[str]] = None, nan_freq: Optional[float] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)

    def extract(self, df: pd.DataFrame) -> 'OrderedString':
        if self.categories is None:
            if isinstance(df[self.name].dtype, pd.CategoricalDtype) and df[self.name].cat.ordered:
                self.categories = df[self.name].cat.categories.to_list()
            else:
                raise ExtractionError
        super().extract(df=df)

        return self

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        return d

    def less_than(self, x: str, y: str) -> bool:
        return self.categories.index(x) < self.categories.index(y)
