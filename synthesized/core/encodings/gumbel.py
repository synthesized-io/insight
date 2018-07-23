import tensorflow as tf

from .encoding import Encoding
from ..transformations import DenseTransformation


class GumbelVariationalEncoding(Encoding):

    def __init__(
        self, name, input_size, encoding_size, num_categories=32, temperature=1.0, hard=False,
        beta=5.0
    ):
        super().__init__(name=name, input_size=input_size, encoding_size=encoding_size)
        self.num_categories = num_categories
        self.temperature = temperature
        self.hard = hard
        self.beta = beta

        self.logits = self.add_module(
            module=DenseTransformation, name='logits', input_size=self.input_size,
            output_size=(self.encoding_size * self.num_categories), batchnorm=False,
            activation='none'
        )

    def specification(self):
        spec = super().specification()
        spec.update(
            num_categories=self.num_categories, temperature=self.temperature, hard=self.hard,
            beta=self.beta
        )
        return spec

    def size(self):
        return self.encoding_size * self.num_categories

    def tf_encode(self, x, encoding_loss=False):
        assert x.shape[1].value == self.encoding_size
        logits = self.logits.transform(x=x)
        logits = tf.reshape(tensor=logits, shape=(-1, self.num_categories))
        probs = tf.nn.softmax(logits=logits, axis=1)
        log_probs = tf.log(x=tf.maximum(x=probs, y=1e-6))
        x = tf.random_uniform(
            shape=tf.shape(input=logits), minval=0.0, maxval=1.0, dtype=tf.float32, seed=None
        )
        x = -tf.log(x=tf.maximum(x=x, y=1e-6))
        x = -tf.log(x=tf.maximum(x=x, y=1e-6))
        x = tf.nn.softmax(logits=((logits + x) / self.temperature), axis=1)
        x = tf.reshape(
            tensor=x, shape=(-1, self.encoding_size * self.num_categories)
        )  # shape should be rank 3
        if self.hard:
            hard = tf.argmax(input=x, axis=2)
            hard = tf.one_hot(
                indices=hard, depth=self.num_categories, on_value=None, off_value=None, axis=None,
                dtype=tf.float32
            )
            # y_hard = tf.cast(
            #     x=tf.equal(
            #         x=encoded, y=tf.reduce_max(
            #             input_tensor=encoded, axis=2, keep_dims=True
            #         )
            #     ), dtype=tf.float32
            # )
            x = tf.stop_gradient(input=(hard - x)) + x
        if encoding_loss:
            encoding_loss = probs * (log_probs - tf.log(x=(1.0 / self.num_categories)))
            encoding_loss = tf.reduce_sum(input_tensor=encoding_loss, axis=(0, 1), keepdims=False)
            # elbo=tf.reduce_sum(p_x.log_prob(x),1) - KL
            encoding_loss *= self.beta
            tf.losses.add_loss(loss=encoding_loss, loss_collection=tf.GraphKeys.LOSSES)
        return x

    def tf_sample(self, n):
        x = tf.random_uniform(
            shape=(n, self.encoding_size * self.num_categories), minval=0.0, maxval=1.0,
            dtype=tf.float32, seed=None
        )  # shape should be rank 3
        x = -tf.log(x=tf.maximum(x=x, y=1e-6))
        x = -tf.log(x=tf.maximum(x=x, y=1e-6))
        if self.hard:
            x = tf.argmax(input=x, axis=2)
            x = tf.one_hot(
                indices=x, depth=self.num_categories, on_value=None, off_value=None, axis=None,
                dtype=tf.float32
            )
        return x
