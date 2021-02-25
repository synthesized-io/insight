from dataclasses import asdict
from typing import Dict, Optional, Sequence

import pandas as pd

from .categorical import String
from ...config import BankLabels


class Bank(String):
    """
    Bank meta.
    """
    dtype = 'U'

    def __init__(
            self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, labels=BankLabels()
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)
        self._params = {k: v for k, v in asdict(labels).items() if v is not None}
        self.children = [String(name) for name in self._params.values()]

    @property
    def params(self) -> Dict[str, Optional[str]]:
        return self._params

    @property
    def labels(self) -> BankLabels:
        return BankLabels(**self.params)

    def extract(self, df: pd.DataFrame):
        super().extract(df)
        return self

    def convert_df_for_children(self, df: pd.DataFrame):

        if not self._is_folded(df):
            return df

        sr_bank = df[self.name]
        index = 0
        if self.labels.bic_label is not None:
            df[self.labels.bic_label] = sr_bank.str.slice(index, index + 4)
            index += 4

        if self.labels.sort_code_label is not None:
            df[self.labels.sort_code_label] = sr_bank.str.slice(index, index + 6)
            index += 6

        if self.labels.account_label is not None:
            df[self.labels.account_label] = sr_bank.str.slice(index, index + 8)
            index += 8

        df.drop(columns=self.name, inplace=True)

    def revert_df_from_children(self, df):

        if self._is_folded(df):
            return df

        df[self.name] = ''
        df[self.name] = df[self.name].str.cat([df[k].astype('string') for k in self.keys()])

        df.drop(
            columns=[k for k in self.keys()],
            inplace=True
        )

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "_params": self.params
        })

        return d
