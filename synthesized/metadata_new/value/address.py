import logging
from dataclasses import astuple
from typing import Dict, Sequence

import pandas as pd

from .categorical import String
from ...config import AddressParams

logger = logging.getLogger(__name__)


class Address(String):
    """
    AddressMeta
    Attributes:
        address_params: a dataclass containing attributes pertaining to addresses
        address_labels: a list of column labels for each attribute, available in
                        self.address_params but stored here for convenience
    """

    def __init__(self, name, categories: Sequence[str] = None, nan_freq: float = None,
                 num_rows: int = None, address_params: AddressParams = AddressParams()):

        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

        self.address_params = address_params
        self.address_labels = [label for label in astuple(address_params) if label is not None]
        self.children = [
            String(label) for label in self.address_labels
        ]

    def extract(self, df: pd.DataFrame):
        super().extract(df=df)
        return self

    def expand(self, df: pd.DataFrame):
        if "collapsed_address" not in df.columns:
            return
        sr_collapsed_address = df["collapsed_address"]
        df[self.address_labels] = sr_collapsed_address.str.split("|", expand=True)

        df.drop(columns="collapsed_address", inplace=True)

    def collapse(self, df: pd.DataFrame):
        df["collapsed_address"] = df[self.address_labels[0]].str.cat(df[self.address_labels[1:]], sep="|")
        df = df.drop(columns=self.address_labels, inplace=True)

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({label: meta.to_dict() for label, meta in self.items()})

        return d
