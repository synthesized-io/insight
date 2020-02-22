from typing import Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp

from .encoding import Encoding
from ..transformations import GaussianTransformation
from ..module import tensorflow_name_scoped


class VariationalLSTMEncoding(Encoding):
    """Encoding for LSTM mode 1

    """
    def __init__(self, input_size, encoding_size, condition_size=0, beta=1.0, lstm_layers=1,
                 name='variational_lstm_encoding'):
        super().__init__(
            name=name, input_size=input_size, encoding_size=encoding_size,
            condition_size=condition_size
        )
        self.beta = beta
        self.lstm_layers = lstm_layers

        self.gaussian = GaussianTransformation(input_size=self.input_size,  output_size=self.encoding_size)
        self.lstm_0 = tf.keras.layers.LSTM(units=self.encoding_size, return_sequences=True)
        self.lstm_i = tf.keras.models.Sequential()
        for _ in range(1, self.lstm_layers):
            self.lstm_i.add(tf.keras.layers.LSTM(units=self.encoding_size, return_sequences=True))

    def build(self, input_shape):
        self.gaussian.build(input_shape)
        self.lstm_0.build((None, None, self.encoding_size))
        self.lstm_i.build((None, None, self.encoding_size))
        self.built = True

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    @tensorflow_name_scoped
    def call(self, inputs, identifier=None, condition=(), return_encoding=False,
             dropout=0.) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        mean, stddev = self.gaussian(inputs)

        e = tf.random.normal(
            shape=tf.shape(input=mean), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        encoding = mean + stddev * e

        if identifier is not None:
            h0, c0 = tf.split(identifier, [self.input_size, self.input_size], axis=0)
            h0 = tf.expand_dims(input=h0, axis=0)
            c0 = tf.expand_dims(input=c0, axis=0)
            identifier = [h0, c0]

        if dropout > 0:
            encoding = tf.nn.dropout(encoding, rate=dropout)
        y = self.lstm_i(
            self.lstm_0(tf.expand_dims(tf.nn.dropout(encoding, rate=dropout), axis=0), initial_state=identifier)
        )
        y = tf.squeeze(y, axis=0)

        kl_loss = self.diagonal_normal_kl_divergence(mu_1=mean, stddev_1=stddev)
        kl_loss = self.beta * kl_loss

        self.add_loss(kl_loss, inputs=inputs)
        tf.summary.histogram(name='mean', data=mean),
        tf.summary.histogram(name='stddev', data=stddev),
        tf.summary.histogram(name='posterior_distribution', data=encoding),
        tf.summary.image(
            name='latent_space_correlation',
            data=tf.abs(tf.reshape(tfp.stats.correlation(encoding),
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
            h0, c0 = tf.split(identifier, [self.input_size, self.input_size], axis=0)
            identifier = [tf.expand_dims(input=h0, axis=0), tf.expand_dims(input=c0, axis=0)]

        e = tf.random.normal(
            shape=(1, n, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        z = self.lstm_i(self.lstm_0(e, initial_state=identifier))

        return tf.squeeze(z, axis=0)

    @property
    def regularization_losses(self):
        return [loss for layer in [self.mean, self.stddev] for loss in layer.regularization_losses]
