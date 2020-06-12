from typing import Optional, Dict

import pandas as pd

from .value_meta import ValueMeta


class IdentifierValue(ValueMeta):
    def __init__(
        self, name, identifiers=None, capacity=None, embedding_size=None
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

        self.capacity = capacity
        self.embedding_size = embedding_size
        if embedding_size is None:
            self.embedding_size = self.capacity
        else:
            self.embedding_size = embedding_size

        self.embeddings = None
        self.placeholder = None
        # self.current_identifier = None

    def __str__(self):
        string = super().__str__()
        string += '{}-{}'.format(self.num_identifiers, self.embedding_size)
        return string

    def specification(self):
        spec = super().specification()
        spec.update(identifiers=self.identifiers, embedding_size=self.embedding_size)
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

    def preprocess(self, df: pd.DataFrame):
        df.loc[:, self.name] = df.loc[:, self.name].map(self.identifier2idx)
        if df.loc[:, self.name].dtype != 'int64':
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='int64')
        return super().preprocess(df)

    def postprocess(self, df: pd.DataFrame):
        df = super().postprocess(df=df)
        df.loc[:, self.name] = df.loc[:, self.name].map(self.idx2identifier)
        return df
