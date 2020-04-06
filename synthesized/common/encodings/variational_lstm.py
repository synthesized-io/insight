from typing import Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp

from .encoding import Encoding
from ..module import tensorflow_name_scoped
from ..transformations import GaussianTransformation


class VariationalLSTMEncoding(Encoding):
    """Encoding for LSTM mode 1"""
    def __init__(self, input_size: int, encoding_size: int, condition_size: int = 0, beta: float = 1.0,
                 lstm_layers: int = 2, name: str = 'lstm'):
        super().__init__(
            name=name, input_size=input_size, encoding_size=encoding_size,
            condition_size=condition_size
        )
        self.beta = beta
        self.lstm_layers = lstm_layers

        self.gaussian = GaussianTransformation(input_size=self.input_size,  output_size=self.encoding_size)

        self.lstm = tf.keras.layers.RNN(
            cell=[tf.keras.layers.LSTMCell(units=self.encoding_size) for _ in range(self.lstm_layers)],
            return_sequences=True, return_state=False)

    def build(self, input_shape):
        self.gaussian.build(input_shape)
        self.lstm.build((None, None, self.encoding_size))
        self.built = True

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    @tensorflow_name_scoped
    def call(self, inputs, identifier=None, condition=(), return_encoding=False, series_dropout=0.,
             n_forecast=0) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:

        mean, stddev = self.gaussian(inputs)
        e = tf.random.normal(
            shape=tf.shape(input=mean), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        encoding = mean + stddev * e

        if identifier is not None:
            c0 = tf.zeros(shape=tf.shape(identifier))
            identifier = [[identifier, c0]] * self.lstm_layers

        lstm_input = encoding
        if series_dropout > 0:
            lstm_input = tf.nn.dropout(lstm_input, rate=series_dropout)

        if n_forecast > 0:
            e = tf.random.normal(shape=(1, n_forecast, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32)
            lstm_input = tf.concat((lstm_input, e), axis=1)

        y = self.lstm(lstm_input, initial_state=identifier)

        kl_loss = self.diagonal_normal_kl_divergence(mu_1=mean, stddev_1=stddev)
        kl_loss = self.beta * kl_loss

        self.add_loss(kl_loss, inputs=inputs)
        tf.summary.histogram(name='mean', data=mean),
        tf.summary.histogram(name='stddev', data=stddev),
        tf.summary.histogram(name='posterior_distribution', data=encoding),
        tf.summary.image(
            name='latent_space_correlation',
            data=tf.abs(tf.reshape(tfp.stats.correlation(tf.reduce_mean(encoding, axis=0)),
                                   shape=(1, self.encoding_size, self.encoding_size, 1)))
        )

        if return_encoding:
            return y, encoding
        else:
            return y

    @tf.function
    @tensorflow_name_scoped
    def sample(self, n, condition=(), identifier=None):
        if identifier is not None:
            h0 = tf.expand_dims(input=identifier, axis=0)
            c0 = tf.zeros(shape=tf.shape(h0))
            identifier = [[h0, c0]] * self.lstm_layers

        e = tf.random.normal(
            shape=(1, n, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        z = self.lstm(e, initial_state=identifier)

        return z

    @property
    def regularization_losses(self):
        return [loss for layer in [self.gaussian] for loss in layer.regularization_losses]
