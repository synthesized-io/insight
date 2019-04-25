from math import log

import pandas as pd
from scipy.stats import beta, gamma, gilbrat, gumbel_r, kstest, lognorm, norm, uniform, weibull_min
import tensorflow as tf
import tensorflow_probability as tfp

from .value import Value
from ..module import Module, tensorflow_name_scoped
import numpy as np


OUTLIERS_PERCENTILE = 0.01
FITTING_SUBSAMPLE = 10000
FITTING_INF_TOLERANCE = 0.01
FITTING_THRESHOLD = 0.1
DISTRIBUTIONS = dict(
    # beta=(beta, tfp.distributions.Beta),
    gamma=(gamma, tfp.distributions.Gamma),
    gilbrat=(gilbrat, None),
    gumbel=(gumbel_r, tfp.distributions.Gumbel),
    log_normal=(lognorm, tfp.distributions.LogNormal),
    uniform=(uniform, tfp.distributions.Uniform),
    weibull=(weibull_min, None)
)


class ContinuousValue(Value):

    def __init__(self, name, distribution=None, distribution_params=None, integer=None, positive=None, nonnegative=None):
        super().__init__(name=name)

        assert distribution is None or distribution == 'normal' or distribution in DISTRIBUTIONS
        self.distribution = distribution
        self.distribution_params = distribution_params
        self.integer = integer
        self.positive = positive
        self.nonnegative = nonnegative

        self.pd_types = ('f', 'i')
        self.pd_cast = (lambda x: pd.to_numeric(x, errors='coerce'))

    def __str__(self):
        string = super().__str__()
        if self.distribution is None:
            string += '-raw'
        else:
            string += '-' + self.distribution
        if self.integer:
            string += '-integer'
        if self.positive:
            string += '-positive'
        elif self.nonnegative:
            string += '-nonnegative'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(
            distribution=self.distribution, distribution_params=self.distribution_params,
            integer=self.integer, positive=self.positive, nonnegative=self.nonnegative
        )
        return spec

    def input_size(self):
        return 1

    def output_size(self):
        return 1

    def placeholders(self):
        yield self.placeholder

    def extract(self, data):
        if data[self.name].dtype.kind not in ('f', 'i'):
            data.loc[:, self.name] = pd.to_numeric(data[self.name], errors='coerce')

        if self.integer is None:
            self.integer = (data[self.name].dtype.kind == 'i')
        elif self.integer and data[self.name].dtype.kind != 'i':
            raise NotImplementedError

        data.loc[:, self.name] = data[self.name].astype(dtype='float32')
        assert not data[self.name].isna().any()
        assert (data[self.name] != float('inf')).all() and (data[self.name] != float('-inf')).all()

        if self.positive is None:
            self.positive = (data[self.name] > 0.0).all()
        elif self.positive and (data[self.name] <= 0.0).all():
            raise NotImplementedError

        if self.nonnegative is None:
            self.nonnegative = (data[self.name] >= 0.0).all()
        elif self.nonnegative and (data[self.name] < 0.0).all():
            raise NotImplementedError

        if self.distribution is not None:
            assert self.distribution_params is not None
            return

        # positive / nonnegative transformation
        full = data[self.name]
        if self.positive or self.nonnegative:
            if self.nonnegative and not self.positive:
                full = np.maximum(full, 0.001)
            # reversed_softplus = np.log(np.maximum((np.exp(full) - 1.0), 1e-6))
            # full = np.where((full < 10.0), reversed_softplus, full)
            full = np.log(np.sign(full) * (1.0 - np.exp(-np.abs(full)))) + np.maximum(full, 0.0)

        # remove outliers
        percentile = [OUTLIERS_PERCENTILE * 50.0, 100.0 - OUTLIERS_PERCENTILE * 50.0]
        lower, upper = np.percentile(full, percentile)
        cleaned_outliers = full[(full >= lower) & (full <= upper)]
        assert len(cleaned_outliers) >= 2
        subsample = np.random.choice(cleaned_outliers, min(len(cleaned_outliers), FITTING_SUBSAMPLE))

        # normal distribution
        self.distribution = 'normal'
        self.distribution_params = mean, stddev = (subsample.mean(), subsample.std())
        transformed = (data[self.name] - mean) / stddev
        min_distance, _ = kstest(transformed, 'norm')

        # other distributions
        for name, (distribution, _) in DISTRIBUTIONS.items():
            distribution_params = distribution.fit(subsample)
            transformed = norm.ppf(distribution.cdf(subsample, *distribution_params))

            assert not np.isnan(transformed).any()
            num_inf = (transformed == float('inf')).sum() + (transformed == float('-inf')).sum()
            if num_inf / len(transformed) > FITTING_INF_TOLERANCE:
                # print('INF TOLERANCE:', name, num_inf / len(transformed))
                continue

            distance, _ = kstest(transformed, 'norm')
            if distance < min_distance:
                # print('extract fit:', name, num_inf / len(transformed))
                min_distance = distance
                self.distribution = name
                self.distribution_params = distribution_params

        # if distance > FITTING_THRESHOLD:
        #     self.distribution = None
        #     self.distribution_params = None

    def preprocess(self, data):
        data = super().preprocess(data=data)
        # TODO: mb removal makes learning more stable (?), an investigation required
        # data = ContinuousValue.remove_outliers(data, self.name, REMOVE_OUTLIERS_PCT)

        if data[self.name].dtype.kind not in ('f', 'i'):
            data.loc[:, self.name] = pd.to_numeric(data[self.name], errors='coerce')

        data.loc[:, self.name] = data[self.name].astype(dtype='float32')
        assert not data[self.name].isna().any()
        assert (data[self.name] != float('inf')).all() and (data[self.name] != float('-inf')).all()

        if self.positive or self.nonnegative:
            if self.nonnegative and not self.positive:
                data.loc[:, self.name] = np.maximum(data[self.name], 0.001)
            # reversed_softplus = np.log(np.maximum((np.exp(data[self.name]) - 1.0), 1e-6))
            # data.loc[:, self.name] = np.where((data[self.name] < 10.0), reversed_softplus, data[self.name])
            data.loc[:, self.name] = np.log(np.sign(data[self.name]) * (1.0 - np.exp(-np.abs(data[self.name])))) + np.maximum(data[self.name], 0.0)

        if self.distribution == 'normal':
            mean, stddev = self.distribution_params
            data.loc[:, self.name] = (data[self.name] - mean) / stddev

        elif self.distribution is not None:
            data.loc[:, self.name] = norm.ppf(
                DISTRIBUTIONS[self.distribution][0].cdf(data[self.name], *self.distribution_params)
            )
            data = data[(data[self.name] != float('inf')) & (data[self.name] != float('-inf'))]

        assert not data[self.name].isna().any()
        assert (data[self.name] != float('inf')).all() and (data[self.name] != float('-inf')).all()

        return data

    def postprocess(self, data):
        if self.distribution == 'normal':
            mean, stddev = self.distribution_params
            data.loc[:, self.name] = data[self.name] * stddev + mean

        elif self.distribution is not None:
            data.loc[:, self.name] = DISTRIBUTIONS[self.distribution][0].ppf(
                norm.cdf(data[self.name]), *self.distribution_params
            )

        if self.positive or self.nonnegative:
            data.loc[:, self.name] = np.log(1 + np.exp(-np.abs(data[self.name]))) + np.maximum(data[self.name], 0.0)
            # np.log(np.exp(data[self.name]) + 1.0)
            if self.nonnegative and not self.positive:
                zeros = np.zeros_like(data[self.name])
                data.loc[:, self.name] = np.where((data[self.name] >= 0.001), data[self.name], zeros)

        assert not data[self.name].isna().any()
        assert (data[self.name] != float('inf')).all() and (data[self.name] != float('-inf')).all()

        if self.integer:
            data.loc[:, self.name] = data[self.name].astype(dtype='int32')

        return data

    def features(self, x=None):
        features = super().features(x=x)
        if x is None:
            features[self.name] = tf.FixedLenFeature(
                shape=(), dtype=tf.float32, default_value=None
            )
        else:
            features[self.name] = tf.train.Feature(
                float_list=tf.train.FloatList(value=(x[self.name],))
            )
        return features

    def module_initialize(self):
        super().module_initialize()
        self.placeholder = tf.placeholder(dtype=tf.float32, shape=(None,), name='input')
        assert self.name not in Module.placeholders
        Module.placeholders[self.name] = self.placeholder

    @tensorflow_name_scoped
    def input_tensor(self, feed=None):
        x = self.placeholder if feed is None else feed[self.name]
        x = tf.expand_dims(input=x, axis=1)
        return x

    @tensorflow_name_scoped
    def output_tensors(self, x):
        x = tf.squeeze(input=x, axis=1)
        return {self.name: x}

    @tensorflow_name_scoped
    def loss(self, x, feed=None, mask=None):
        target = self.input_tensor(feed=feed)[:, :1]  # first value since date adds more information
        if mask is not None:
            target = tf.boolean_mask(tensor=target, mask=mask)
            x = tf.boolean_mask(tensor=x, mask=mask)
        # loss = tf.nn.l2_loss(t=(target - x))
        loss = tf.losses.mean_squared_error(
            labels=target, predictions=x, weights=1.0, scope=None,
            loss_collection=tf.GraphKeys.LOSSES
        )  # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        return loss

    @tensorflow_name_scoped
    def distribution_loss(self, samples):
        samples = tf.squeeze(input=samples, axis=1)

        if self.distribution == 'gamma':
            shape, location, scale = self.distribution_params
            distribution = DISTRIBUTIONS[self.distribution][1](concentration=shape, rate=1.0)
            samples = tf.where(
                condition=(samples < location), x=(samples + 2 * location), y=samples
            )
            samples = (samples - location) / scale
            samples = distribution.cdf(value=samples) / scale
        elif self.distribution == 'gumbel':
            location, scale = self.distribution_params
            distribution = DISTRIBUTIONS[self.distribution][1](location=location, scale=scale)
            samples = tf.where(
                condition=(samples < location), x=(samples + 2 * location), y=samples
            )
            samples = distribution.cdf(value=samples)
        elif self.distribution == 'log_normal':
            shape, location, scale = self.distribution_params
            distribution = DISTRIBUTIONS[self.distribution][1](loc=log(scale), scale=scale)
            samples = tf.where(
                condition=(samples < location), x=(samples + 2 * location), y=samples
            )
            samples = samples - location
            samples = distribution.cdf(value=samples)
        elif self.distribution == 'weibull':
            shape, location, scale = self.distribution_params
            distribution = DISTRIBUTIONS['gamma'][1](concentration=shape, rate=1.0)
            samples = tf.where(
                condition=(samples < location), x=(samples + 2 * location), y=samples
            )
            samples = (samples - location) / scale
            samples = distribution.cdf(value=samples) / scale
        else:
            assert False

        samples = tf.boolean_mask(tensor=samples, mask=tf.math.logical_not(x=tf.is_nan(x=samples)))
        normal_distribution = tfp.distributions.Normal(loc=0.0, scale=1.0)
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
        num_samples = tf.cast(x=tf.shape(input=samples)[0], dtype=tf.float32)
        # jarque_bera = num_samples / 6.0 * (tf.square(x=skewness) + \
        #     0.25 * tf.square(x=(kurtosis - 3.0)))
        jarque_bera = tf.square(x=skewness) + tf.square(x=(kurtosis - 3.0))
        jarque_bera_loss = tf.squared_difference(x=jarque_bera, y=0.0)
        loss = mean_loss + variance_loss + jarque_bera_loss

        return loss
