from math import log
import tensorflow as tf
from synthesized.core.values import Value


class ContinuousValue(Value):

    def __init__(self, name, positive=False):
        super().__init__(name=name)
        self.positive = positive

    def size(self):
        return 1

    def _initialize(self):
        self.placeholder = tf.placeholder(dtype=tf.float32, shape=(None,), name='input')

    def input_tensor(self):
        tensor = tf.expand_dims(input=self.placeholder, axis=1, name=None)
        return tensor

    def output_tensor(self, x):
        if self.positive:
            x = tf.nn.softplus(features=x, name=None)
        x = tf.squeeze(input=x, axis=1, name=None)
        return x

    def loss(self, x):
        if self.positive:
            x = tf.nn.softplus(features=x, name=None)
        target = self.input_tensor()
        loss = tf.losses.mean_squared_error(
            labels=target, predictions=x, weights=1.0, scope=None,
            loss_collection=tf.GraphKeys.LOSSES
        )  # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        return loss
