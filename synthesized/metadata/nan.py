import numpy as np
import pandas as pd

from .continuous import ContinuousMeta
from .value_meta import ValueMeta


class NanMeta(ValueMeta):

    def __init__(
        self, name: str, value: ValueMeta, produce_nans: bool = False
    ):
        super().__init__(name=name)

        assert isinstance(value, ContinuousMeta)
        # assert isinstance(value, (CategoricalValue, ContinuousValue))
        self.value = value

        self.embedding_initialization = 'orthogonal-small'

        self.produce_nans = produce_nans

    def __str__(self):
        string = super().__str__()
        string += '-' + str(self.value)
        return string

    def specification(self):
        spec = super().specification()
        spec.update(
            value=self.value.specification(), produce_nans=self.produce_nans
        )
        return spec

    def learned_input_columns(self):
        return self.value.learned_input_columns()

    def learned_output_columns(self):
        return self.value.learned_output_columns()

    def extract(self, df):
        column = df[self.value.name]
        if column.dtype.kind not in self.value.pd_types:
            column = self.value.pd_cast(column)
        df_clean = df[column.notna()]
        self.value.extract(df=df_clean)

    def preprocess(self, df):
        df.loc[:, self.value.name] = pd.to_numeric(df.loc[:, self.value.name], errors='coerce')

        nan = df.loc[:, self.value.name].isna()
        if sum(~nan) > 0:
            df.loc[~nan, :] = self.value.preprocess(df=df.loc[~nan, :])
        df.loc[:, self.value.name] = df.loc[:, self.value.name].astype(np.float32)

        return super().preprocess(df=df)

    def postprocess(self, df):
        df = super().postprocess(df=df)

        nan = df.loc[:, self.value.name].isna()
        df.loc[~nan, :] = self.value.postprocess(df=df.loc[~nan, :])

        return df
