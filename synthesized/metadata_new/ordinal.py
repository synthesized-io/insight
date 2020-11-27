from typing import Optional, List, cast, Dict

import pandas as pd

from .base import Ordinal, Domain
from .exceptions import MetaNotExtractedError, ExtractionError


class OrderedString(Ordinal[str]):
    class_name: str = 'String'
    dtype: str = 'str'

    def __init__(
            self, name: str, domain: Optional[Domain[str]] = None, nan_freq: Optional[float] = None,
            order: Optional[List[str]] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq)
        self.order = order

    def extract(self, df: pd.DataFrame) -> 'OrderedString':
        if self.order is None:
            if isinstance(df[self.name].dtype, pd.CategoricalDtype) and df[self.name].cat.ordered:
                self.order = df[self.name].cat.categories.to_list()
            else:
                raise ExtractionError
        super().extract(df=df)

        return self

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "order": self.order
        })
        return d

    def less_than(self, x: str, y: str) -> bool:
        if not self._extracted:
            raise MetaNotExtractedError

        self.domain = cast(Domain[str], self.domain)
        self.order = cast(List[str], self.order)

        if x not in self.domain or y not in self.domain:
            raise ValueError(f"x={x} or y={y} are not valid categories.")

        return self.order.index(x) < self.order.index(y)
