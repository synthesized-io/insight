from dataclasses import asdict
from typing import Dict, Optional, Sequence

import pandas as pd

from .categorical import String
from ..base import ValueMeta
from ...config import BankLabels


class Bank(String):
    """
    Bank meta.
    """
    dtype = 'U'

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, labels=BankLabels()
    ):
        self._params = {k: v for k, v in asdict(labels).items() if v is not None}
        children = [
            String(name)
            for name in self._params.values() if name is not None
        ] if children is None else children
        super().__init__(name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

    @property
    def params(self) -> Dict[str, str]:
        return self._params

    @property
    def labels(self) -> BankLabels:
        return BankLabels(**self.params)

    def extract(self, df: pd.DataFrame):
        super().extract(df)
        return self

    def convert_df_for_children(self, df: pd.DataFrame):

        col_index = df.columns.get_loc(self.name)
        sr_bank = df[self.name]
        index = 0
        if self.labels.bic_label is not None:
            df.insert(col_index, self.labels.bic_label, sr_bank.str.slice(index, index + 4))
            col_index += 1
            index += 4

        if self.labels.sort_code_label is not None:
            df.insert(col_index, self.labels.sort_code_label, sr_bank.str.slice(index, index + 6))
            col_index += 1
            index += 6

        if self.labels.account_label is not None:
            df.insert(col_index, self.labels.account_label, sr_bank.str.slice(index, index + 8))

        df.drop(columns=self.name, inplace=True)

    def revert_df_from_children(self, df):
        col_index = df.columns.get_loc(list(self.keys())[0])
        df.insert(col_index, self.name, '')
        df[self.name] = df[self.name].str.cat([df[k].astype('string') for k in self.keys()])

        df.drop(columns=list(self.keys()), inplace=True)

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "_params": self.params
        })

        return d
