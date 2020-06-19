from typing import Optional, Dict, List

import pandas as pd

from .value_meta import ValueMeta


class IdentifierMeta(ValueMeta):
    def __init__(
        self, name, identifiers=None
    ):
        super().__init__(name=name)

        if identifiers is None:
            self.identifiers = None
            self.num_identifiers = None
        elif isinstance(identifiers, int):
            self.identifiers = self.num_identifiers = identifiers
        else:
            self.identifiers = sorted(identifiers)
            self.num_identifiers = len(self.identifiers)

        self.identifier2idx: Optional[Dict] = None

        self.placeholder = None
        # self.current_identifier = None

    def __str__(self):
        string = super().__str__()
        return string

    def specification(self):
        spec = super().specification()
        spec.update(identifiers=self.identifiers)
        return spec

    def extract(self, df):
        super().extract(df=df)

        if self.identifiers is None:
            self.identifiers = sorted(df.loc[:, self.name].unique())
            self.num_identifiers = len(self.identifiers)
        elif sorted(df.loc[:, self.name].unique()) != self.identifiers:
            raise NotImplementedError

        self.identifier2idx = {k: i for i, k in enumerate(self.identifiers)}
        self.idx2identifier = {i: k for i, k in enumerate(self.identifiers)}

    def learned_input_columns(self) -> List[str]:
        return []

    def learned_output_columns(self) -> List[str]:
        return []

    def set_index(self, df: pd.DataFrame):
        if df.index.names == [None]:
            df.set_index(self.name, inplace=True)
        else:
            df.set_index(self.name, inplace=True, append=True)

    def preprocess(self, df: pd.DataFrame):
        df.loc[:, self.name] = df.loc[:, self.name].map(self.identifier2idx)
        if df.loc[:, self.name].dtype != 'int64':
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='int64')
        return super().preprocess(df)

    def postprocess(self, df: pd.DataFrame):
        df = super().postprocess(df=df)
        df.loc[:, self.name] = df.loc[:, self.name].map(self.idx2identifier)
        return df
