from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer
from tensorflow_probability import distributions as tfd

from .value import Value
from ..module import Module, tensorflow_name_scoped


class ContinuousValue(Value):

    def __init__(
        self, name: str, weight: float,
        # Scenario
        integer: bool = None, positive: bool = None, nonnegative: bool = None,
        transformer_n_quantiles=1000
    ):
        super().__init__(name=name)

        self.weight = weight

        self.integer = integer
        self.positive = positive
        self.nonnegative = nonnegative
        self.distribution: Optional[str] = None
        self.distribution_params: Optional[Tuple[Any, ...]] = None
        self.transformer = QuantileTransformer(n_quantiles=transformer_n_quantiles, output_distribution='normal')

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

        if column.dtype.kind not in ('f', 'i'):
            column = self.pd_cast(column)

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

        if self.distribution is not None:
            assert self.distribution_params is not None
            return

        if column.nunique() == 1:
            self.distribution = 'dirac'
            self.distribution_params = (column[0],)
            return

        column = column.values
        # positive / nonnegative transformation
        if self.positive or self.nonnegative:
            if self.nonnegative and not self.positive:
                column = np.maximum(column, 0.001)
            column = np.log(np.sign(column) * (1.0 - np.exp(-np.abs(column)))) + np.maximum(column, 0.0)

        self.transformer.fit(column.reshape(-1, 1))

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: mb removal makes learning more stable (?), an investigation required
        # df = ContinuousValue.remove_outliers(df, self.name, REMOVE_OUTLIERS_PCT)

        if df[self.name].dtype.kind not in ('f', 'i'):
            df.loc[:, self.name] = self.pd_cast(df[self.name])

        df.loc[:, self.name] = df[self.name].astype(dtype='float32')
        assert not df[self.name].isna().any()
        assert (df[self.name] != float('inf')).all() and (df[self.name] != float('-inf')).all()

        if self.distribution == 'dirac':
            return df

        if self.positive or self.nonnegative:
            if self.nonnegative and not self.positive:
                df.loc[:, self.name] = np.maximum(df[self.name], 0.001)
            df.loc[:, self.name] = np.maximum(df[self.name], 0.0) + np.log(
                np.sign(df[self.name]) * (1.0 - np.exp(-np.abs(df[self.name])))
            )

        df.loc[:, self.name] = self.transformer.transform(df[self.name].values.reshape(-1, 1))

        assert not df[self.name].isna().any()
        assert (df[self.name] != float('inf')).all() and (df[self.name] != float('-inf')).all()

        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        if self.distribution == 'dirac':
            assert self.distribution_params is not None
            df.loc[:, self.name] = self.distribution_params[0]
        else:
            df.loc[:, self.name] = self.transformer.inverse_transform(df[self.name].values.reshape(-1, 1))
            if self.positive or self.nonnegative:
                df.loc[:, self.name] = np.log(1 + np.exp(-np.abs(df[self.name]))) + \
                                       np.maximum(df[self.name], 0.0)
                if self.nonnegative and not self.positive:
                    zeros = np.zeros_like(df[self.name])
                    df.loc[:, self.name] = np.where(
                        (df[self.name] >= 0.001), df[self.name], zeros
                    )

        assert not df[self.name].isna().any()
        assert (df[self.name] != float('inf')).all() and (df[self.name] != float('-inf')).all()

        if self.integer:
            df.loc[:, self.name] = df[self.name].astype(dtype='int32')

        return df

    def module_initialize(self) -> None:
        super().module_initialize()

        # Input placeholder for value
        self.add_placeholder(name=self.name, dtype=tf.float32, shape=(None,))

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        assert Module.placeholders is not None
        return [Module.placeholders[self.name]]

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

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        assert len(ys) == 1

        if self.distribution is None:
            return tf.constant(value=0.0, dtype=tf.float32)

        samples = ys[0]

        samples = tf.boolean_mask(tensor=samples, mask=tf.math.logical_not(x=tf.is_nan(x=samples)))
        normal_distribution = tfd.Normal(loc=0.0, scale=1.0)
        samples = normal_distribution.quantile(value=samples)
        samples = tf.boolean_mask(tensor=samples, mask=tf.is_finite(x=samples))
        samples = tf.boolean_mask(tensor=samples, mask=tf.math.logical_not(x=tf.is_nan(x=samples)))

        mean, variance = tf.nn.moments(x=samples, axes=0)
        mean_loss = tf.squared_difference(x=mean, y=0.0)
        variance_loss = tf.squared_difference(x=variance, y=1.0)

        mean = tf.stop_gradient(input=tf.reduce_mean(input_tensor=samples, axis=0))
        difference = samples - mean
        squared_difference = tf.square(x=difference)
        variance = tf.reduce_mean(input_tensor=squared_difference, axis=0)
        third_moment = tf.reduce_mean(input_tensor=(squared_difference * difference), axis=0)
        fourth_moment = tf.reduce_mean(input_tensor=tf.square(x=squared_difference), axis=0)
        skewness = third_moment / tf.pow(x=variance, y=1.5)
        kurtosis = fourth_moment / tf.square(x=variance)
        # num_samples = tf.cast(x=tf.shape(input=samples)[0], dtype=tf.float32)
        # jarque_bera = num_samples / 6.0 * (tf.square(x=skewness) + \
        #     0.25 * tf.square(x=(kurtosis - 3.0)))
        jarque_bera = tf.square(x=skewness) + tf.square(x=(kurtosis - 3.0))
        jarque_bera_loss = tf.squared_difference(x=jarque_bera, y=0.0)
        loss = mean_loss + variance_loss + jarque_bera_loss

        return loss
