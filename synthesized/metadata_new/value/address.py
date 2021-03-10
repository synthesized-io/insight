import logging
from dataclasses import asdict
from typing import Dict, Optional, Sequence

import pandas as pd

from .categorical import String
from ..base import ValueMeta
from ...config import AddressLabels

logger = logging.getLogger(__name__)


class Address(String):
    """
    Address
    """

    def __init__(
            self, name, children: Optional[Sequence[ValueMeta]] = None, categories: Optional[Sequence[str]] = None,
            nan_freq: Optional[float] = None, num_rows: Optional[int] = None, labels: AddressLabels = AddressLabels()
    ):
        self._params = {k: v for k, v in asdict(labels).items() if v is not None}
        children = [
            String(name)
            for name in self._params.values() if name is not None
        ] if children is None else children
        super().__init__(name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

    @property
    def params(self) -> Dict[str, Optional[str]]:
        return self._params

    @property
    def labels(self) -> AddressLabels:
        return AddressLabels(**self.params)

    def extract(self, df: pd.DataFrame):
        super().extract(df=df)
        return self

    def convert_df_for_children(self, df: pd.DataFrame):
        if self.name not in df.columns:
            raise KeyError
        sr_collapsed_address = df[self.name]
        df[list(self.keys())] = sr_collapsed_address.astype(str).str.split("|", n=len(self.keys()) - 1, expand=True)

        df.drop(columns=self.name, inplace=True)

    def revert_df_from_children(self, df: pd.DataFrame):
        df[self.name] = df[list(self.keys())[0]].astype(str).str.cat(
            [df[k].astype(str) for k in list(self.keys())[1:]], sep="|", na_rep=''
        )
        df.drop(columns=list(self.keys()), inplace=True)

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "_params": self.params
        })
        return d
