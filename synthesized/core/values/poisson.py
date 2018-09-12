import tensorflow as tf

from .value import Value
from ..module import Module


class PoissonValue(Value):

    def __init__(self, name, avgnum=None):
        super().__init__(name=name)
        self.avgnum = avgnum

    def specification(self):
        spec = super().specification()
        spec.update(avgnum=self.avgnum)
        return spec

    def input_size(self):
        return 1

    def trainable_labels(self):
        yield self.name

    def placeholders(self):
        yield self.placeholder

    def extract(self, data):
        if self.avgnum is None:
            self.avgnum = (data[self.name].mean() + data[self.name].std()) / 2.0

    def preprocess(self, data):
        data.loc[:, self.name] = data[self.name].astype(dtype='int64')
        return data

    def feature(self, x=None):
        if x is None:
            return tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None)
        else:
            return tf.train.Feature(float_list=tf.train.Int64List(value=(x,)))

    def tf_initialize(self):
        super().tf_initialize()
        self.placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='input')
        Module.placeholders[self.name] = self.placeholder

    def tf_input_tensor(self, feed=None):
        x = self.placeholder if feed is None else feed
        x = (tf.to_float(x=x) - self.avgnum) / self.avgnum
        x = tf.expand_dims(input=x, axis=1)
        return x

    def tf_output_tensors(self, x):
        x = tf.squeeze(input=x, axis=1)
        x = x * self.avgnum + self.avgnum
        x = tf.to_int64(x=x)
        return {self.name: x}

    def tf_loss(self, x, feed=None):
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
