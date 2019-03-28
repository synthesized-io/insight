import tensorflow as tf

from .value import Value
from ..module import Module, tensorflow_name_scoped
import numpy as np

REMOVE_OUTLIERS_PCT = 0.5


class ContinuousValue(Value):

    def __init__(self, name, positive=None, nonnegative=None, integer=None):
        super().__init__(name=name)
        self.positive = positive
        self.nonnegative = nonnegative
        if self.positive is not None:
            self.nonnegative = False
        elif self.nonnegative is not None:
            self.positive = False
        self.integer = integer

    def __str__(self):
        string = super().__str__()
        if self.positive:
            string += '-positive'
        if self.nonnegative:
            string += '-nonnegative'
        if self.integer:
            string += '-integer'
        return string

    def specification(self):
        spec = super().specification()
        spec.update(positive=self.positive, nonnegative=self.nonnegative, integer=self.integer)
        return spec

    def input_size(self):
        return 1

    def placeholders(self):
        yield self.placeholder

    def extract(self, data):
        if self.positive is None:
            self.positive = (data[self.name] > 0.0).all()
        elif self.positive and (data[self.name] <= 0.0).all():
            raise NotImplementedError
        if self.nonnegative is None:
            self.nonnegative = (data[self.name] >= 0.0).all()
        elif self.nonnegative and (data[self.name] < 0.0).all():
            raise NotImplementedError
        if self.integer is None:
            self.integer = (data[self.name].dtype.kind == 'i')
        elif self.integer and data[self.name].dtype.kind != 'i':
            raise NotImplementedError

    def encode(self, data):
        data.loc[:, self.name] = data[self.name].astype(dtype='float32')
        return data

    def preprocess(self, data):
        data = super().preprocess(data)
        # TODO: mb removal makes learning more stable (?), an investigation required
        # data = ContinuousValue.remove_outliers(data, self.name, REMOVE_OUTLIERS_PCT)
        return data

    def postprocess(self, data):
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
        if self.positive or self.nonnegative:
            if self.nonnegative and not self.positive:
                x = tf.maximum(x=x, y=0.001)
            reversed_softplus = tf.log(x=tf.maximum(x=(tf.exp(x=x) - 1.0), y=1e-6))
            x = tf.where(condition=(x < 10.0), x=reversed_softplus, y=x)
        x = tf.expand_dims(input=x, axis=1)
        return x

    @tensorflow_name_scoped
    def output_tensors(self, x):
        x = tf.squeeze(input=x, axis=1)
        if self.positive or self.nonnegative:
            x = tf.nn.softplus(features=x)
            if self.nonnegative and not self.positive:
                zeros = tf.zeros_like(tensor=x, dtype=tf.float32, optimize=True)
                x = tf.where(condition=(x >= 0.001), x=x, y=zeros)
        return {self.name: x}

    @tensorflow_name_scoped
    def loss(self, x, feed=None):
        # if self.positive:                      ??????????????????????????????????????
        #     x = tf.nn.softplus(features=x)
        target = self.input_tensor(feed=feed)
        # target = tf.Print(target, (x, target))

        # relative = tf.maximum(x=x, y=target) / (tf.minimum(x=x, y=target) + 1.0)
        # # relative = tf.Print(relative, (relative,))
        # relative = tf.minimum(x=relative, y=10.0)
        # relative = tf.squeeze(input=relative, axis=1)
        # loss = tf.reduce_mean(input_tensor=(relative - 1.0), axis=0, keepdims=False)
        # loss = tf.losses.add_loss(loss=loss, loss_collection=tf.GraphKeys.LOSSES)
        loss = tf.losses.mean_squared_error(
            labels=target, predictions=x, weights=1.0, scope=None,
            loss_collection=tf.GraphKeys.LOSSES
        )  # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        return loss

    @tensorflow_name_scoped
    def distribution_loss(self, samples):
        return 0.0

    @staticmethod
    def remove_outliers(data, name, pct):
        percentiles = [pct / 2., 100 - pct / 2.]
        start, end = np.percentile(data[name], percentiles)
        data = data[data[name] != float('nan')]
        data = data[data[name] != float('inf')]
        data = data[data[name] != float('-inf')]
        return data[(data[name] > start) & (data[name] < end)]

    @tensorflow_name_scoped
    def distribution_loss(self, samples):
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
