from typing import Dict, Optional, Sequence, cast

import pandas as pd

from ..base import Ordinal, ValueMeta
from ..exceptions import ExtractionError


class OrderedString(Ordinal[str]):
    dtype: str = 'U'

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None
    ):
        super().__init__(name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

    def extract(self, df: pd.DataFrame) -> 'OrderedString':
        if self._categories is None:
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
        return cast(Sequence, self.categories).index(x) < cast(Sequence, self.categories).index(y)
