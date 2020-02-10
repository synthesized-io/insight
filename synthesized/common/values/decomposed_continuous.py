from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import lfilter
import tensorflow as tf

from .value import Value
from . import ContinuousValue
from ..module import tensorflow_name_scoped


class DecomposedContinuousValue(Value):

    def __init__(
        self, name: str, weight: float,
        # Scenario
        integer: bool = None, float: bool = True, positive: bool = None, nonnegative: bool = None,
        distribution: str = None, distribution_params: Tuple[Any, ...] = None,
        transformer_n_quantiles: int = 1000, transformer_noise: Optional[float] = 1e-7,
        A: float = 0, B: float = 0, C: float = 0
        # A: float = None, B: float = None, C: float = None
    ):
        super().__init__(name=name)

        self.weight = weight
        self.low_freq_weight = 2.
        self.high_freq_weight = 1.

        assert (A is None and B is None and C is None) or (A is not None and B is not None and C is not None)
        self.A = A
        self.B = B
        self.C = C

        continuous_kwargs = dict()
        continuous_kwargs['weight'] = weight
        continuous_kwargs['integer'] = integer
        continuous_kwargs['float'] = float
        continuous_kwargs['positive'] = positive
        continuous_kwargs['nonnegative'] = nonnegative
        continuous_kwargs['distribution'] = distribution
        continuous_kwargs['distribution_params'] = distribution_params
        continuous_kwargs['use_quantile_transformation'] = False
        continuous_kwargs['transformer_n_quantiles'] = transformer_n_quantiles
        continuous_kwargs['transformer_noise'] = transformer_noise

        self.low_freq_value = ContinuousValue(name=(self.name + '-low-freq'), **continuous_kwargs)
        self.high_freq_value = ContinuousValue(name=(self.name + '-high-freq'), **continuous_kwargs)

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

        a, b, k, y_low, y_high = _decompose_signal(column, A=self.A, B=self.B, C=self.C)
        self.A = a if self.A is None else self.A
        self.B = b if self.B is None else self.B
        self.C = k if self.C is None else self.C
        df[self.name + '-low-freq'] = y_low
        df[self.name + '-high-freq'] = y_high

        self.low_freq_value.extract(df)
        self.high_freq_value.extract(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        column = df.loc[:, self.name]
        _, _, _, y_low, y_high = _decompose_signal(column, A=self.A, B=self.B, C=self.C)
        df[self.name + '-low-freq'] = y_low
        df[self.name + '-high-freq'] = y_high

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
        x = np.array(range(len(df)))

        df[self.name] = self.A * (x ** 2) + self.B * x + self.C + y_low + y_high

        return df.drop([self.low_freq_value.name, self.high_freq_value.name], axis=1)

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


def _decompose_signal(y, A=None, B=None, C=None):
    y = np.array(y)
    len_y = len(y)

    x = np.array(range(len_y))
    x_2 = x ** 2

    if A is None or B is None or C is None:
        assert A is None and B is None and C is None
        xx = np.vstack([x_2, x, np.ones(len_y)]).T
        A, B, C = np.linalg.lstsq(xx, y, rcond=None)[0]

    y1 = y - (A * x_2 + B * x + C)

    b_n = int(len(y) / 100)
    b = [1. / b_n] * b_n
    a = 1
    y_low = lfilter(b, a, y1)
    y_high = y1 - y_low

    return A, B, C, y_low, y_high

