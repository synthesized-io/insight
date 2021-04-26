from math import log
from typing import Any, Dict, Sequence

import tensorflow as tf
from scipy.stats import gamma, gilbrat, gumbel_r, lognorm, uniform, weibull_min
from tensorflow_probability import distributions as tfd

from .value import Value
from ..module import tensorflow_name_scoped
from ...config import ContinuousConfig

DISTRIBUTIONS = dict(
    gamma=(gamma, tfd.Gamma),
    gilbrat=(gilbrat, None),
    gumbel=(gumbel_r, tfd.Gumbel),
    log_normal=(lognorm, tfd.LogNormal),
    uniform=(uniform, tfd.Uniform),
    weibull=(weibull_min, None)
)


class ContinuousValue(Value):

    def __init__(
        self, name: str, config: ContinuousConfig = ContinuousConfig()
    ):
        super().__init__(name=name)
        self.weight = config.continuous_weight

    def specification(self) -> Dict[str, Any]:
        spec = super().specification()
        spec.update(
            weight=self.weight
        )
        return spec

    def learned_input_size(self) -> int:
        return 1

    def learned_output_size(self) -> int:
        return 1

    @tensorflow_name_scoped
    def unify_inputs(self, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        assert len(xs) == 1
        nan_mask = tf.math.is_nan(xs[0])
        x = tf.where(nan_mask, tf.random.normal(xs[0].shape), xs[0])
        x = tf.expand_dims(x, axis=-1)
        return x

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor, **kwargs) -> Sequence[tf.Tensor]:
        y = tf.reshape(y, shape=(-1,))
        return (y,)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: Sequence[tf.Tensor]) -> tf.Tensor:
        nan_mask = tf.math.is_nan(xs[0])
        x = tf.where(nan_mask, tf.squeeze(y), xs[0])
        target = tf.expand_dims(x, axis=-1)

        loss = tf.squeeze(input=tf.math.squared_difference(x=y, y=target), axis=-1)
        loss = self.weight * tf.reduce_mean(input_tensor=loss, axis=None)
        tf.summary.scalar(name=self.name, data=loss)
        return loss

    @tensorflow_name_scoped
    def distribution_loss(self, ys: Sequence[tf.Tensor]) -> tf.Tensor:
        assert len(ys) == 1

        if self.distribution is None:
            return tf.constant(value=0.0, dtype=tf.float32)

        samples = ys[0]

        if self.distribution == 'normal':
            assert self.distribution_params is not None
            loc, scale = self.distribution_params
            distribution = tfd.Normal(loc=loc, scale=scale)
            samples = distribution.cdf(value=samples)
        elif self.distribution == 'gamma':
            assert self.distribution_params is not None
            shape, location, scale = self.distribution_params
            distribution_class = DISTRIBUTIONS[self.distribution][1]
            assert distribution_class is not None
            distribution = distribution_class(concentration=shape, rate=1.0)
            samples = tf.where(
                condition=(samples < location), x=(samples + 2 * location), y=samples
            )
            samples = (samples - location) / scale
            samples = distribution.cdf(value=samples) / scale
        elif self.distribution == 'gumbel':
            assert self.distribution_params is not None
            location, scale = self.distribution_params
            distribution_class = DISTRIBUTIONS[self.distribution][1]
            assert distribution_class is not None
            distribution = distribution_class(location=location, scale=scale)
            samples = tf.where(
                condition=(samples < location), x=(samples + 2 * location), y=samples
            )
            samples = distribution.cdf(value=samples)
        elif self.distribution == 'log_normal':
            assert self.distribution_params is not None
            shape, location, scale = self.distribution_params
            distribution_class = DISTRIBUTIONS[self.distribution][1]
            assert distribution_class is not None
            distribution = distribution_class(loc=log(scale), scale=scale)
            samples = tf.where(
                condition=(samples < location), x=(samples + 2 * location), y=samples
            )
            samples = samples - location
            samples = distribution.cdf(value=samples)
        elif self.distribution == 'weibull':
            assert self.distribution_params is not None
            shape, location, scale = self.distribution_params
            distribution_class = DISTRIBUTIONS['gamma'][1]
            assert distribution_class is not None
            distribution = distribution_class(concentration=shape, rate=1.0)
            samples = tf.where(
                condition=(samples < location), x=(samples + 2 * location), y=samples
            )
            samples = (samples - location) / scale
            samples = distribution.cdf(value=samples) / scale
        else:
            assert False

        samples = tf.boolean_mask(tensor=samples, mask=tf.math.logical_not(x=tf.math.is_nan(x=samples)))
        normal_distribution = tfd.Normal(loc=0.0, scale=1.0)
        samples = normal_distribution.quantile(value=samples)
        samples = tf.boolean_mask(tensor=samples, mask=tf.math.is_finite(x=samples))
        samples = tf.boolean_mask(tensor=samples, mask=tf.math.logical_not(x=tf.math.is_nan(x=samples)))

        mean, variance = tf.nn.moments(x=samples, axes=0)
        mean_loss = tf.math.squared_difference(x=mean, y=0.0)
        variance_loss = tf.math.squared_difference(x=variance, y=1.0)

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
        jarque_bera_loss = tf.math.squared_difference(x=jarque_bera, y=0.0)
        loss = mean_loss + variance_loss + jarque_bera_loss

        return loss
