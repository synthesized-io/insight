from typing import Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp

from .encoding import Encoding
from ..transformations import GaussianTransformation
from ..module import tensorflow_name_scoped


class VariationalRecurrentEncoding(Encoding):
    """Encoding for LSTM mode 2

    Original paper: https://arxiv.org/pdf/1412.6581.pdf
    """

    def __init__(self, input_size, encoding_size, condition_size=0, beta=1., name='variational_lstm2_encoding'):
        super().__init__(
            name=name, input_size=input_size, encoding_size=encoding_size,
            condition_size=condition_size
        )
        self.beta = beta

        self.lstm_encoder = tf.keras.layers.LSTM(units=encoding_size, return_state=True)
        self.gaussian = GaussianTransformation(input_size=self.encoding_size, output_size=self.encoding_size)
        self.lstm_decoder = tf.keras.layers.LSTM(units=encoding_size, return_sequences=True, return_state=True)

    def build(self, input_shape):
        self.lstm_encoder.build(input_shape=(None, None, self.input_size))
        self.gaussian.build(input_shape=(None, self.encoding_size))
        self.lstm_decoder.build(input_shape=(None, None, self.encoding_size))
        self.built = True

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    def call(self, inputs, identifier=None, condition=(), return_encoding=False,
             series_dropout=0.) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:

        if series_dropout > 0.:
            inputs = tf.nn.dropout(inputs, rate=series_dropout)
        _, h_out, _ = self.lstm_encoder(inputs)

        mean, stddev = self.gaussian(h_out)
        e_h = tf.random.normal(shape=tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)
        encoding_h = mean + stddev * e_h

        encoding_c = tf.zeros(shape=tf.shape(mean), dtype=tf.float32)
        encoding = [encoding_h, encoding_c]

        # input_decoder = tf.concat((tf.zeros((1, 1, self.encoding_size)), inputs[:, 1:-1, :]), axis=1)
        # input_decoder = tf.nn.dropout(input_decoder, rate=self.dropout)

        # input_decoder = tf.expand_dims(tf.tile(encoding_h, [tf.shape(inputs)[1], 1]), axis=0)
        # encoding = None
        input_decoder = tf.zeros(tf.shape(inputs))
        # if dropout > 0.:
        #     input_decoder = tf.nn.dropout(input_decoder, rate=dropout)

        y, _, _ = self.lstm_decoder(input_decoder, initial_state=encoding)

        # y = self.lstm_loop(n=tf.shape(inputs)[1], h_i=encoding_h)

        kl_loss = self.diagonal_normal_kl_divergence(mu_1=mean, stddev_1=stddev)
        # kl_loss *= self.beta * self.increase_beta_multiplier(t_start=250, t_end=350)
        kl_loss *= self.beta

        self.add_loss(kl_loss, inputs=inputs)

        tf.summary.scalar(name='kl-loss', data=kl_loss)
        tf.summary.histogram(name='mean', data=mean),
        tf.summary.histogram(name='stddev', data=stddev),
        tf.summary.histogram(name='posterior_distribution', data=encoding_h),
        tf.summary.image(
            name='latent_space_correlation',
            data=tf.abs(tf.reshape(tfp.stats.correlation(encoding_h),
                                   shape=(1, self.encoding_size, self.encoding_size, 1)))
        )

        if return_encoding:
            return y, encoding_h
        else:
            return y

    # @tf.function
    # @tensorflow_name_scoped
    # def sample(self, n, condition=()):
    #
    #     n = tf.cast(n, dtype=tf.int32)
    #     h_i = tf.random.normal(shape=(1, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32)
    #     c_i = tf.zeros(shape=(1, self.encoding_size))
    #
    #     y = tf.TensorArray(dtype=tf.float32, size=n, clear_after_read=True)
    #
    #     y_i = tf.zeros(shape=(1, 1, self.encoding_size))
    #
    #     for i in tf.range(n):
    #         state_i = [h_i, c_i]
    #         y_i, h_i, c_i = self.lstm_decoder(y_i, initial_state=state_i)
    #         y = y.write(i, tf.squeeze(y_i))
    #
    #     z = y.stack()
    #     return z

    @tf.function
    @tensorflow_name_scoped
    def sample(self, n, condition=(), identifier=None):

        h_i = tf.random.normal(shape=(1, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32)
        c_i = tf.zeros(shape=(1, self.encoding_size))

        input_decoder = tf.zeros(shape=(1, n, self.encoding_size))
        y, _, _ = self.lstm_decoder(input_decoder, initial_state=[h_i, c_i])

        # input_decoder = tf.expand_dims(tf.tile(h_i, [n, 1]), axis=0)
        # y, _, _ = self.lstm_decoder(input_decoder, initial_state=None)

        return tf.squeeze(y, axis=0)

    @tf.function
    @tensorflow_name_scoped
    def lstm_loop(self, n, h_i):

        c_i = tf.zeros(shape=(1, self.encoding_size))

        y = tf.TensorArray(dtype=tf.float32, size=n, clear_after_read=True)
        y_i = tf.zeros(shape=(1, 1, self.encoding_size))

        for i in tf.range(n):
            state_i = [h_i, c_i]
            y_i, h_i, c_i = self.lstm_decoder(y_i, initial_state=state_i)
            y = y.write(i, tf.squeeze(y_i))

        z = y.stack()
        return z

    @property
    def regularization_losses(self):
        return [loss for layer in [self.gaussian] for loss in layer.regularization_losses]
