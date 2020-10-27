from typing import List

import numpy as np
import pandas as pd

from .continuous import ContinuousMeta
from .value_meta import ValueMeta
from .date import DateMeta


class NanMeta(ValueMeta):

    def __init__(
        self, name: str, value: ValueMeta
    ):
        super().__init__(name=name)

        assert isinstance(value, ContinuousMeta)
        # assert isinstance(value, (CategoricalValue, ContinuousValue))
        self.value = value

        self.embedding_initialization = 'orthogonal-small'

    def __str__(self):
        string = super().__str__()
        string += '-' + str(self.value)
        return string

    def specification(self):
        spec = super().specification()
        spec.update(
            value=self.value.specification()
        )
        return spec

    def columns(self) -> List[str]:
        return self.value.columns()

    def learned_input_columns(self) -> List[str]:
        return self.value.learned_input_columns()

    def learned_output_columns(self) -> List[str]:
        return self.value.learned_output_columns()

    def extract(self, df):
        super().extract(df=df)

        column = df[self.value.name]
        if column.dtype.kind not in self.value.pd_types:
            if type(self.value) is DateMeta:
                column = self.value.to_datetime(column)
            elif type(self.value) is ContinuousMeta:
                column = self.value.pd_cast(column)
            else:
                raise TypeError(f"{type(self.value)} is not supported by NanMeta")

        try:
            df_clean = df[(~column.isin([np.Inf, -np.Inf])) & column.notna()]
        except OverflowError:
            # comparing date to inf throws an OverflowError
            # catch exception when column is datetime
            df_clean = df[column.notna()]

        self.value.extract(df=df_clean)

    def preprocess(self, df):

        if type(self.value) is DateMeta:
            df = self.value.preprocess(df)
            # convert NaT to NaN
            df.loc[df[self.value.name].isna(), self.value.name] = np.nan

        elif isinstance(self.value, ContinuousMeta):
            df.loc[:, self.value.name] = pd.to_numeric(df.loc[:, self.value.name], errors='coerce')
            nan_inf = df.loc[:, self.value.name].isna() | df.loc[:, self.value.name].isin([np.Inf, -np.Inf])

            if sum(~nan_inf) > 0:
                df.loc[~nan_inf, :] = self.value.preprocess(df=df.loc[~nan_inf, :])


        else:
            raise TypeError(f"{type(self.value)} is not supported by NanMeta")

        df.loc[:, self.value.name] = df.loc[:, self.value.name].astype(np.float32)
        return super().preprocess(df=df)

    def postprocess(self, df):
        df = super().postprocess(df=df)

        nan_inf = df.loc[:, self.value.name].isna() | df.loc[:, self.value.name].isin([np.Inf, -np.Inf])
        df.loc[~nan_inf, :] = self.value.postprocess(df=df.loc[~nan_inf, :])

        self.set_dtypes(df)
        return df
