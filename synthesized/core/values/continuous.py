import tensorflow as tf

from .value import Value
from ..module import Module


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

    def trainable_labels(self):
        yield self.name

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

    def preprocess(self, data):
        data.loc[:, self.name] = data[self.name].astype(dtype='float32')
        return data

    def postprocess(self, data):
        if self.integer:
            data.loc[:, self.name] = data[self.name].astype(dtype='int32')
        return data

    def feature(self, x=None):
        if x is None:
            return tf.FixedLenFeature(shape=(), dtype=tf.float32, default_value=None)
        else:
            return tf.train.Feature(float_list=tf.train.FloatList(value=(x,)))

    def tf_initialize(self):
        super().tf_initialize()
        self.placeholder = tf.placeholder(dtype=tf.float32, shape=(None,), name='input')
        assert self.name not in Module.placeholders
        Module.placeholders[self.name] = self.placeholder

    def tf_input_tensor(self, feed=None):
        x = self.placeholder if feed is None else feed
        if self.positive or self.nonnegative:
            if self.nonnegative and not self.positive:
                x = tf.maximum(x=x, y=0.001)
            reversed_softplus = tf.log(x=tf.maximum(x=(tf.exp(x=x) - 1.0), y=1e-6))
            x = tf.where(condition=(x < 10.0), x=reversed_softplus, y=x)
        x = tf.expand_dims(input=x, axis=1)
        return x

    def tf_output_tensors(self, x):
        x = tf.squeeze(input=x, axis=1)
        if self.positive or self.nonnegative:
            x = tf.nn.softplus(features=x)
            if self.nonnegative and not self.positive:
                zeros = tf.zeros_like(tensor=x, dtype=tf.float32, optimize=True)
                x = tf.where(condition=(x >= 0.001), x=x, y=zeros)
        return {self.name: x}

    def tf_loss(self, x, feed=None):
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
