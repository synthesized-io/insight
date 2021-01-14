from typing import Dict, Optional, Sequence

import pandas as pd

from .categorical import String


class Bank(String):
    """
    Bank meta.
    """
    dtype = 'U'

    def __init__(
            self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, bic_name: Optional[str] = None, sort_code_name: Optional[str] = None,
            account_name: Optional[str] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

        self.bic_name = bic_name
        self.sort_code_name = sort_code_name
        self.account_name = account_name

        self.children = [
            String(child_name) for child_name in [bic_name, sort_code_name, account_name] if child_name is not None
        ]

    def extract(self, df: pd.DataFrame):
        super().extract(df)  # call super here so we can get max, min from datetime.

        return self

    def expand(self, df: pd.DataFrame):

        sr_bank = df[self.name]
        index = 0
        if self.bic_name is not None:
            df[self.bic_name] = sr_bank.str.slice(index, index + 4)
            index += 4

        if self.sort_code_name is not None:
            df[self.sort_code_name] = sr_bank.str.slice(index, index + 6)
            index += 6

        if self.account_name is not None:
            df[self.account_name] = sr_bank.str.slice(index, index + 8)
            index += 8

        df.drop(columns=self.name, inplace=True)

    def collapse(self, df):
        df[self.name] = ''
        df[self.name].str.cat(self.keys())

        df.drop(
            columns=self.keys(),
            inplace=True
        )

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "bic_name": self.bic_name,
            "sort_code_name": self.sort_code_name,
            "account_name": self.account_name
        })

        return d
