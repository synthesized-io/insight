import logging
from dataclasses import asdict
from typing import Dict, Optional, Sequence, Type, cast

import pandas as pd

from .categorical import String
from ..base import Meta
from ...config import AddressLabels

logger = logging.getLogger(__name__)


class Address(String):
    """
    Address
    """

    def __init__(
            self, name, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, labels: AddressLabels = AddressLabels()
    ):

        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)
        self._params = {k: v for k, v in asdict(labels).items() if v is not None}

        if len(self.params.values()) == 0:
            raise ValueError("At least one of labels must be given")

        if name in self.params.values():
            raise ValueError("Value of 'name' can't be equal to any other label.")

        self.children = [
            String(name)
            for name in self._params.values() if name is not None
        ]

    @property
    def params(self) -> Dict[str, str]:
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

    @classmethod
    def from_dict(cls: Type['Address'], d: Dict[str, object]) -> 'Address':
        name = cast(str, d["name"])
        d.pop("class_name", None)
        params = cast(Dict[str, str], d.pop("_params"))
        labels = AddressLabels(**params)

        extracted = d.pop("extracted", False)
        children = cast(Dict[str, Dict[str, object]], d.pop("children")) if "children" in d else None

        meta = cls(name=name, labels=labels)
        for attr, value in d.items():
            setattr(meta, attr, value)

        setattr(meta, '_extracted', extracted)

        if children is not None:
            meta_children = []
            for child in children.values():
                class_name = cast(str, child['class_name'])
                meta_children.append(Meta.from_name_and_dict(class_name, child))

            meta.children = meta_children

        return meta
