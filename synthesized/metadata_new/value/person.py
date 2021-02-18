from dataclasses import asdict, fields
from typing import Dict, Optional, Sequence

import pandas as pd

from .categorical import String
from ...config import PersonLabels


class Person(String):
    """Person meta."""

    def __init__(
            self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, labels: Optional[PersonLabels] = None,
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)
        self._labels = labels if labels is not None else PersonLabels()
        self.children = [
            String(getattr(self._labels, label.name))
            for label in fields(self._labels) if getattr(self._labels, label.name) is not None
        ]

    @property
    def labels(self) -> Dict[str, Optional[str]]:
        return asdict(self._labels)

    def extract(self, df: pd.DataFrame):
        super().extract(df)

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
            "_labels": self.labels
        })

        return d
