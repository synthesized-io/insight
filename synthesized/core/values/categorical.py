import tensorflow as tf
from synthesized.core.values import Value


class CategoricalValue(Value):

    def __init__(self, name, num_categories, smoothing=0.25):
        super().__init__(name=name)
        self.num_categories = num_categories
        self.smoothing = smoothing

    def size(self):
        return self.num_categories

    def _initialize(self):
        self.placeholder = tf.placeholder(dtype=tf.int64, shape=(None,), name='input')

    def input_tensor(self):
        tensor = tf.one_hot(
            indices=self.placeholder, depth=self.num_categories, on_value=1.0, off_value=0.0,
            axis=1, dtype=tf.float32, name=None
        )
        return tensor

    def output_tensor(self, x):
        x = tf.argmax(input=x, axis=1, name=None)
        return x

    def loss(self, x):
        target = self.input_tensor()
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=target, logits=x, weights=1.0, label_smoothing=self.smoothing, scope=None,
            loss_collection=tf.GraphKeys.LOSSES
        )  # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        return loss
