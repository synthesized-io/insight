from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import filtfilt

from .continuous import ContinuousMeta
from .value_meta import ValueMeta


class DecomposedContinuousMeta(ValueMeta):

    def __init__(
        self, name: str, weight: float, identifier: Optional[str],
        # Scenario
        integer: bool = None, float: bool = True, positive: bool = None, nonnegative: bool = None,
        distribution: str = None, distribution_params: Tuple[Any, ...] = None,
        use_quantile_transformation: bool = False,
        transformer_n_quantiles: int = 1000, transformer_noise: Optional[float] = 1e-7,
        low_freq_weight: float = 1., high_freq_weight: float = 1.
    ):
        super().__init__(name=name)

        self.weight = weight
        self.identifier = identifier
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight

        self.weight = weight
        self.integer = integer
        self.float = float
        self.positive = positive
        self.nonnegative = nonnegative
        self.use_quantile_transformation = use_quantile_transformation
        self.transformer_n_quantiles = transformer_n_quantiles
        self.transformer_noise = transformer_noise

        continuous_kwargs: Dict[str, Any] = dict()
        continuous_kwargs['weight'] = weight
        continuous_kwargs['use_quantile_transformation'] = use_quantile_transformation
        continuous_kwargs['transformer_n_quantiles'] = transformer_n_quantiles
        continuous_kwargs['transformer_noise'] = transformer_noise

        self.low_freq_value = ContinuousMeta(name=(self.name + '-low-freq'), **continuous_kwargs)
        self.high_freq_value = ContinuousMeta(name=(self.name + '-high-freq'), **continuous_kwargs)

        self.pd_cast = (lambda x: pd.to_numeric(x, errors='coerce', downcast='integer'))

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            weight=self.weight, low_freq=self.low_freq_value.specification(),
            high_freq=self.high_freq_value.specification()
        )
        return spec

    def learned_input_columns(self) -> List[str]:
        columns = list()
        columns.extend(self.low_freq_value.learned_input_columns())
        columns.extend(self.high_freq_value.learned_input_columns())

        return columns

    def learned_output_columns(self) -> List[str]:
        columns = list()
        columns.extend(self.low_freq_value.learned_output_columns())
        columns.extend(self.high_freq_value.learned_output_columns())

        return columns

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)

        if df.loc[:, self.name].dtype.kind not in ('f', 'i'):
            df.loc[:, self.name] = self.pd_cast(df.loc[:, self.name])

        self.float = (df.loc[:, self.name].dtype.kind == 'f')

        if self.integer is None:
            self.integer = (df.loc[:, self.name].dtype.kind == 'i') or \
                df.loc[:, self.name].apply(lambda x: x.is_integer()).all()
        elif self.integer and df.loc[:, self.name].dtype.kind != 'i':
            raise NotImplementedError

        df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='float32')

        if self.positive is None:
            self.positive = (df.loc[:, self.name] > 0.0).all()
        elif self.positive and (df.loc[:, self.name] <= 0.0).all():
            raise NotImplementedError

        if self.nonnegative is None:
            self.nonnegative = (df.loc[:, self.name] >= 0.0).all()
        elif self.nonnegative and (df.loc[:, self.name] < 0.0).all():
            raise NotImplementedError

        df = _decompose_df(df, column_name=self.name, identifier=self.identifier)

        self.low_freq_value.extract(df)
        self.high_freq_value.extract(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:

        if df.loc[:, self.name].dtype.kind not in ('f', 'i'):
            df.loc[:, self.name] = self.pd_cast(df.loc[:, self.name])

        assert not df.loc[:, self.name].isna().any()
        assert (df.loc[:, self.name] != float('inf')).all() and (df.loc[:, self.name] != float('-inf')).all()

        df.loc[:, self.name] = df.loc[:, self.name].astype(np.float32)

        df = _decompose_df(df, column_name=self.name, identifier=self.identifier)

        self.low_freq_value.preprocess(df)
        self.high_freq_value.preprocess(df)

        return df.drop([self.name], axis=1)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        df = self.low_freq_value.postprocess(df)
        df = self.high_freq_value.postprocess(df)

        y_low = np.array(df[self.low_freq_value.name])
        y_high = np.array(df[self.high_freq_value.name])

        df[self.name] = y_low + y_high
        df.drop([self.low_freq_value.name, self.high_freq_value.name], axis=1, inplace=True)

        if self.nonnegative:
            df.loc[(df.loc[:, self.name] < 0.001), self.name] = 0

        assert not df.loc[:, self.name].isna().any()
        assert (df.loc[:, self.name] != float('inf')).all() and (df.loc[:, self.name] != float('-inf')).all()

        if self.integer:
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='int32')

        if self.float and df.loc[:, self.name].dtype != 'float32':
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='float32')

        return df


def _decompose_df(df, column_name, identifier=None):
    df = df.copy()
    df[column_name + '-low-freq'] = 0

    if identifier is not None:
        def decompose_signal_df(d):
            d.loc[:, column_name + '-low-freq'] = _decompose_signal(d.loc[:, column_name])
            return d
        df = df.groupby(identifier).apply(decompose_signal_df)
    else:
        df.loc[:, column_name + '-low-freq'] = _decompose_signal(df.loc[:, column_name])
    df.loc[:, column_name + '-high-freq'] = df.loc[:, column_name] - df.loc[:, column_name + '-low-freq']
    return df


def _decompose_signal(y):
    y = np.array(y)

    b_n = int(max(np.ceil(len(y) / 100), 2))
    b = [1. / b_n] * b_n
    a = 1
    pad_len = 3 * len(b)
    if len(y) > pad_len:
        return filtfilt(b, a, y)
    else:
        return np.zeros(len(y))
