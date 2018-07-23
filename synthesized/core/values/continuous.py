import tensorflow as tf

from .value import Value


class ContinuousValue(Value):

    def __init__(self, name, positive=False):
        super().__init__(name=name)
        self.positive = positive

    def specification(self):
        spec = super().specification()
        spec.update(positive=self.positive)
        return spec

    def input_size(self):
        return 1

    def feature(self, x=None):
        if x is None:
            return tf.FixedLenFeature(shape=(), dtype=tf.float32, default_value=None)
        else:
            return tf.train.Feature(float_list=tf.train.FloatList(value=(x,)))

    def tf_initialize(self):
        super().tf_initialize()
        self.placeholder = tf.placeholder(dtype=tf.float32, shape=(None,), name='input')

    def tf_input_tensor(self, feed=None):
        x = self.placeholder if feed is None else feed
        x = tf.expand_dims(input=x, axis=1)
        return x

    def tf_output_tensor(self, x):
        if self.positive:
            x = tf.nn.softplus(features=x)
        x = tf.squeeze(input=x, axis=1)
        return x

    def tf_loss(self, x, feed=None):
        if self.positive:
            x = tf.nn.softplus(features=x)
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
