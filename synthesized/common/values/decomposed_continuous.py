from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import filtfilt
import tensorflow as tf

from .value import Value
from . import ContinuousValue
from ..module import tensorflow_name_scoped


class DecomposedContinuousValue(Value):

    def __init__(
        self, name: str, weight: float, identifier: Optional[str],
        # Scenario
        integer: bool = None, float: bool = True, positive: bool = None, nonnegative: bool = None,
        distribution: str = None, distribution_params: Tuple[Any, ...] = None,
        use_quantile_transformation: bool = False,
        transformer_n_quantiles: int = 1000, transformer_noise: Optional[float] = 1e-7
    ):
        super().__init__(name=name)

        self.weight = weight
        self.identifier = identifier
        self.low_freq_weight = 2.
        self.high_freq_weight = 1.

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

        self.low_freq_value = ContinuousValue(name=(self.name + '-low-freq'), **continuous_kwargs)
        self.high_freq_value = ContinuousValue(name=(self.name + '-high-freq'), **continuous_kwargs)

        self.pd_cast = (lambda x: pd.to_numeric(x, errors='coerce', downcast='integer'))

    def __str__(self) -> str:
        string = super().__str__()
        return string

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            weight=self.weight, low_freq=self.low_freq_value.specification(),
            high_freq=self.high_freq_value.specification()
        )
        return spec

    def learned_input_size(self) -> int:
        return self.low_freq_value.learned_input_size() + self.high_freq_value.learned_input_size()

    def learned_output_size(self) -> int:
        return self.low_freq_value.learned_output_size() + self.high_freq_value.learned_output_size()

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

    @tensorflow_name_scoped
    def build(self):
        if not self.built:
            self.low_freq_value.build()
            self.high_freq_value.build()

        self.built = True

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)
        column = df.loc[:, self.name]

        if column.dtype.kind not in ('f', 'i'):
            column = self.pd_cast(column)

        self.float = (column.dtype.kind == 'f')

        if self.integer is None:
            self.integer = (column.dtype.kind == 'i') or column.apply(lambda x: x.is_integer()).all()
        elif self.integer and column.dtype.kind != 'i':
            raise NotImplementedError

        column = column.astype(dtype='float32')

        if self.positive is None:
            self.positive = (column > 0.0).all()
        elif self.positive and (column <= 0.0).all():
            raise NotImplementedError

        if self.nonnegative is None:
            self.nonnegative = (column >= 0.0).all()
        elif self.nonnegative and (column < 0.0).all():
            raise NotImplementedError

        df = _decompose_df(df, column_name=self.name, identifier=self.identifier)

        self.low_freq_value.extract(df)
        self.high_freq_value.extract(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _decompose_df(df, column_name=self.name, identifier=self.identifier)

        self.low_freq_value.preprocess(df)
        self.high_freq_value.preprocess(df)

        return df.drop([self.name], axis=1)

        # return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        df = self.low_freq_value.postprocess(df)
        df = self.high_freq_value.postprocess(df)

        y_low = np.array(df[self.low_freq_value.name])
        y_high = np.array(df[self.high_freq_value.name])

        df[self.name] = y_low + y_high
        df.drop([self.low_freq_value.name, self.high_freq_value.name], axis=1, inplace=True)

        if self.nonnegative and not self.positive:
            df.loc[(df.loc[:, self.name] < 0.001), self.name] = 0

        assert not df.loc[:, self.name].isna().any()
        assert (df.loc[:, self.name] != float('inf')).all() and (df.loc[:, self.name] != float('-inf')).all()

        if self.integer:
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='int32')

        if self.float and df.loc[:, self.name].dtype != 'float32':
            df.loc[:, self.name] = df.loc[:, self.name].astype(dtype='float32')

        return df

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        self.build()
        xs[0] = self.low_freq_value.unify_inputs(xs=xs[0:1])
        xs[1] = self.high_freq_value.unify_inputs(xs=xs[1:2])
        return tf.concat(values=xs, axis=1)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        return tf.concat((
            self.low_freq_value.output_tensors(y=tf.expand_dims(y[:, 0], axis=1)),
            self.high_freq_value.output_tensors(y=tf.expand_dims(y[:, 1], axis=1))
        ), axis=0)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        loss_low_freq = self.low_freq_value.loss(y=tf.expand_dims(y[:, 0], axis=1), xs=xs[0:1])
        loss_high_freq = self.high_freq_value.loss(y=tf.expand_dims(y[:, 1], axis=1), xs=xs[1:2])

        return self.low_freq_weight * loss_low_freq + self.high_freq_weight * loss_high_freq


# def _decompose_signal(y, A=None, B=None, C=None):
#     y = np.array(y)
#     len_y = len(y)
#
#     x = np.array(range(len_y))
#     x_2 = x ** 2
#
#     if A is None or B is None or C is None:
#         assert A is None and B is None and C is None
#         xx = np.vstack([x_2, x, np.ones(len_y)]).T
#         A, B, C = np.linalg.lstsq(xx, y, rcond=None)[0]
#
#     y1 = y - (A * x_2 + B * x + C)
#
#     b_n = int(len(y) / 100)
#     b = [1. / b_n] * b_n
#     a = 1
#     y_low = lfilter(b, a, y1)
#     y_high = y1 - y_low
#
#     return A, B, C, y_low, y_high


def _decompose_df(df, column_name, identifier=None):
    df = df.copy()
    df[column_name + '-low-freq'] = 0

    if identifier is not None:
        def decompose_signal_df(d):
            d.loc[:, column_name + '-low-freq'] = _decompose_signal(d.loc[:, column_name])
            return d
        df = df.groupby(identifier).apply(decompose_signal_df)
    else:
        df[column_name + '-low-freq'] = _decompose_signal(df[column_name])
    df[column_name + '-high-freq'] = df.loc[:, column_name] - df.loc[:, column_name + '-low-freq']
    return df


def _decompose_signal(y):
    y = np.array(y)

    b_n = int(max(np.ceil(len(y) / 100), 2))
    b = [1. / b_n] * b_n
    a = 1
    y_low = filtfilt(b, a, y)

    return y_low
