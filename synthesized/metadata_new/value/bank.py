from typing import Dict, Optional, Sequence

import pandas as pd

from .categorical import String
from ...config import BankParams


class Bank(String):
    """
    Bank meta.
    """
    dtype = 'U'

    def __init__(
            self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, bic_label: Optional[str] = None, sort_code_label: Optional[str] = None,
            account_label: Optional[str] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

        self.bic_label = bic_label
        self.sort_code_label = sort_code_label
        self.account_label = account_label

        self.children = [
            String(child_label) for child_label in [bic_label, sort_code_label, account_label] if child_label is not None
        ]

    @classmethod
    def from_params(cls, params: BankParams) -> 'Bank':
        bank = Bank(
            name=params.name, bic_label=params.bic_label, sort_code_label=params.sort_code_label,
            account_label=params.account_label
        )

        return bank

    def extract(self, df: pd.DataFrame):
        super().extract(df)

        return self

    def convert_df_for_children(self, df: pd.DataFrame):

        sr_bank = df[self.name]
        index = 0
        if self.bic_label is not None:
            df[self.bic_label] = sr_bank.str.slice(index, index + 4)
            index += 4

        if self.sort_code_label is not None:
            df[self.sort_code_label] = sr_bank.str.slice(index, index + 6)
            index += 6

        if self.account_label is not None:
            df[self.account_label] = sr_bank.str.slice(index, index + 8)
            index += 8

        df.drop(columns=self.name, inplace=True)

    def revert_df_from_children(self, df):
        df[self.name] = ''
        df[self.name] = df[self.name].str.cat([df[k].astype('string') for k in self.keys()])

        df.drop(
            columns=[k for k in self.keys()],
            inplace=True
        )

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "bic_label": self.bic_label,
            "sort_code_label": self.sort_code_label,
            "account_label": self.account_label
        })

        return d
