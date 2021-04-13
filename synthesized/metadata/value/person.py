from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Type, cast

import numpy as np
import pandas as pd

from .categorical import String
from ...config import PersonLabels


class Person(String):
    """Person meta."""

    def __init__(
            self, name: str, children: Optional[Sequence[String]] = None,
            categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, labels: PersonLabels = PersonLabels(),
    ):
        columns = [c for c in labels.__dict__.values() if c is not None]
        if all([c is None for c in columns]):
            raise ValueError("At least one of labels must be given")
        if name in columns:
            raise ValueError("Value of 'name' can't be equal to any other label.")
        if len(columns) > len(np.unique(columns)):
            raise ValueError("There can't be any duplicated labels.")

        self._params = {k: v for k, v in asdict(labels).items() if v is not None}
        children = [
            String(name, num_rows=num_rows, nan_freq=nan_freq)
            for name in self._params.values() if name is not None
        ] if children is None else children
        super().__init__(name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

    @property
    def params(self) -> Dict[str, str]:
        return self._params

    @property
    def labels(self) -> PersonLabels:
        return PersonLabels(**self.params)

    def extract(self, df: pd.DataFrame):
        super().extract(df)
        return self

    def convert_df_for_children(self, df: pd.DataFrame):
        if self.name not in df.columns:
            raise KeyError
        col_index = df.columns.get_loc(self.name)
        sr_collapsed_person = df[self.name]
        df.drop(columns=self.name, inplace=True)
        df_child = sr_collapsed_person.astype(str).str.split("|", n=len(self.keys()) - 1, expand=True)
        for n, col in enumerate(self.keys()):
            df.insert(col_index + n, col, df_child.iloc[:, n])

    def revert_df_from_children(self, df: pd.DataFrame):
        col_index = min([df.columns.get_loc(k) for k in self.keys()])
        sr_collapsed_person = df[list(self.keys())[0]].astype(str).str.cat(
            [df[k].astype(str) for k in list(self.keys())[1:]], sep="|", na_rep=''
        )
        df.drop(columns=list(self.keys()), inplace=True)
        df.insert(col_index, self.name, sr_collapsed_person)

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "_params": self.params
        })
        return d

    @classmethod
    def from_dict(cls: Type['Person'], d: Dict[str, object]) -> 'Person':
        name = cast(str, d["name"])
        d.pop("class_name", None)
        params = cast(Dict[str, str], d.pop("_params"))
        labels = PersonLabels(**params)

        extracted = d.pop("extracted", False)
        children = cast(Dict[str, Dict[str, object]], d.pop("children")) if "children" in d else None
        if children is not None:
            meta_children: List[String] = []
            for child in children.values():
                class_name = cast(str, child['class_name'])
                meta_children.append(String.from_name_and_dict(class_name, child))

        meta = cls(name=name, children=meta_children, labels=labels)
        for attr, value in d.items():
            setattr(meta, attr, value)

        setattr(meta, '_extracted', extracted)

        return meta
