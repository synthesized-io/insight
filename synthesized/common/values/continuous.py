from math import log
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import gamma, gilbrat, gumbel_r, lognorm, norm, uniform, weibull_min
from sklearn.preprocessing import QuantileTransformer
from tensorflow_probability import distributions as tfd

from .value import Value
from ..module import tensorflow_name_scoped

DISTRIBUTIONS = dict(
    # beta=(beta, tfd.Beta),
    gamma=(gamma, tfd.Gamma),
    gilbrat=(gilbrat, None),
    gumbel=(gumbel_r, tfd.Gumbel),
    log_normal=(lognorm, tfd.LogNormal),
    uniform=(uniform, tfd.Uniform),
    weibull=(weibull_min, None)
)


class ContinuousValue(Value):

    def __init__(
        self, name: str, weight: float,
        # Scenario
        integer: bool = None, float: bool = True, positive: bool = None, nonnegative: bool = None,
        distribution: str = None, distribution_params: Tuple[Any, ...] = None,
        transformer_n_quantiles: int = 1000, transformer_noise: Optional[float] = 1e-7
    ):
        super().__init__(name=name)

        self.weight = weight

        self.integer = integer
        self.float = float
        self.positive = positive
        self.nonnegative = nonnegative

        assert distribution is None or distribution == 'normal' or distribution in DISTRIBUTIONS
        self.distribution = distribution
        self.distribution_params = distribution_params

        # transformer is fitted in `extract`
        self.transformer_n_quantiles = transformer_n_quantiles
        self.transformer_noise = transformer_noise
        self.transformer: Optional[QuantileTransformer] = None

        self.pd_types: Tuple[str, ...] = ('f', 'i')
        self.pd_cast = (lambda x: pd.to_numeric(x, errors='coerce', downcast='integer'))

    def __str__(self) -> str:
        string = super().__str__()
        if self.distribution is None:
            string += '-raw'
        else:
            string += '-' + self.distribution
        if self.integer:
            string += '-integer'
        if self.positive and self.distribution != 'dirac':
            string += '-positive'
        elif self.nonnegative and self.distribution != 'dirac':
            string += '-nonnegative'
        return string

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            weight=self.weight, integer=self.integer, positive=self.positive,
            nonnegative=self.nonnegative, distribution=self.distribution,
            distribution_params=self.distribution_params,
        )
        return spec

    def learned_input_size(self) -> int:
        return 1

    def learned_output_size(self) -> int:
        return 1

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)
        column = df[self.name]

        # we allow extraction only if distribution hasn't been set
        assert self.distribution is None

        if column.dtype.kind not in ('f', 'i'):
            column = self.pd_cast(column)

        self.float = (column.dtype.kind == 'f')

        if self.integer is None:
            self.integer = (column.dtype.kind == 'i') or column.apply(lambda x: x.is_integer()).all()
        elif self.integer and column.dtype.kind != 'i':
            raise NotImplementedError

        column = column.astype(dtype='float32')
        assert not column.isna().any()
        assert (column != float('inf')).all() and (column != float('-inf')).all()

        if self.positive is None:
            self.positive = (column > 0.0).all()
        elif self.positive and (column <= 0.0).all():
            raise NotImplementedError

        if self.nonnegative is None:
            self.nonnegative = (column >= 0.0).all()
        elif self.nonnegative and (column < 0.0).all():
            raise NotImplementedError

        column = column.values
        # positive / nonnegative transformation
        if self.positive or self.nonnegative:
            if self.nonnegative and not self.positive:
                column = np.maximum(column, 0.001)
            # column = np.log(np.sign(column) * (1.0 - np.exp(-np.abs(column)))) + np.maximum(column, 0.0)

        if self.transformer_noise:
            column += np.random.normal(scale=self.transformer_noise, size=len(column))

        self.transformer = QuantileTransformer(n_quantiles=self.transformer_n_quantiles, output_distribution='normal')
        self.transformer.fit(column.reshape(-1, 1))

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: mb removal makes learning more stable (?), an investigation required
        # df = ContinuousValue.remove_outliers(df, self.name, REMOVE_OUTLIERS_PCT)

        if df[self.name].dtype.kind not in ('f', 'i'):
            df[self.name] = self.pd_cast(df[self.name])

        assert not df[self.name].isna().any()
        assert (df[self.name] != float('inf')).all() and (df[self.name] != float('-inf')).all()

        if self.distribution == 'dirac':
            return df

        if self.nonnegative and not self.positive:
            df[self.name] = np.maximum(df[self.name], 0.001)

        if self.distribution == 'normal':
            assert self.distribution_params is not None
            mean, stddev = self.distribution_params
            df[self.name] = (df[self.name] - mean) / stddev

        elif self.distribution is not None:
            df[self.name] = norm.ppf(
                DISTRIBUTIONS[self.distribution][0].cdf(df[self.name], *self.distribution_params)
            )
            df = df[(df[self.name] != float('inf')) & (df[self.name] != float('-inf'))]
        elif self.transformer:
            if self.transformer_noise:
                df[self.name] += np.random.normal(scale=self.transformer_noise, size=len(df[self.name]))
            df[self.name] = self.transformer.transform(df[self.name].values.reshape(-1, 1))

        assert not df[self.name].isna().any()
        assert (df[self.name] != float('inf')).all() and (df[self.name] != float('-inf')).all()

        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        if self.distribution == 'dirac':
            assert self.distribution_params is not None
            df[self.name] = self.distribution_params[0]

        else:
            if self.distribution == 'normal':
                assert self.distribution_params is not None
                mean, stddev = self.distribution_params
                df[self.name] = df[self.name] * stddev + mean
            elif self.distribution is not None:
                df[self.name] = DISTRIBUTIONS[self.distribution][0].ppf(
                    norm.cdf(df[self.name]), *self.distribution_params
                )
            elif self.transformer:
                df[self.name] = self.transformer.inverse_transform(df[self.name].values.reshape(-1, 1))

            if self.nonnegative and not self.positive:
                df.loc[(df[self.name] < 0.001), self.name] = 0

        assert not df[self.name].isna().any()
        assert (df[self.name] != float('inf')).all() and (df[self.name] != float('-inf')).all()

        if self.integer:
            df[self.name] = df[self.name].astype(dtype='int32')

        if self.float and df[self.name].dtype != 'float32':
            df[self.name] = df[self.name].astype(dtype='float32')

        return df

    def module_initialize(self) -> None:
        super().module_initialize()

        # Input placeholder for value
        self.placeholder_initialize(dtype=tf.float32, shape=(None,))

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        return [self.placeholder]

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        assert len(xs) == 1
        return tf.expand_dims(input=xs[0], axis=1)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        y = tf.squeeze(input=y, axis=1)
        return [y]

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor], mask: tf.Tensor = None) -> tf.Tensor:
        if self.distribution == 'dirac':
            return tf.constant(value=0.0, dtype=tf.float32)

        assert len(xs) == 1
        target = xs[0]
        target = tf.expand_dims(input=target, axis=1)
        # target = self.input_tensors(xs=xs)[:, :1]  # first value since date adds more information
        if mask is not None:
            target = tf.boolean_mask(tensor=target, mask=mask)
            y = tf.boolean_mask(tensor=y, mask=mask)
        # loss = tf.nn.l2_loss(t=(target - x))
        loss = tf.squeeze(input=tf.math.squared_difference(x=y, y=target), axis=1)
        loss = self.weight * tf.reduce_mean(input_tensor=loss, axis=0)
        return loss
