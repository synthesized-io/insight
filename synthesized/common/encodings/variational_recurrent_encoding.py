from typing import Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp

from .encoding import Encoding
from ..module import tensorflow_name_scoped
from ..transformations import GaussianTransformation


class VariationalRecurrentEncoding(Encoding):
    """Encoding for LSTM mode 2

    Original paper: https://arxiv.org/pdf/1412.6581.pdf
    """

    def __init__(self, input_size: int, encoding_size: int, condition_size: int = 0, beta: float = 1.0,
                 lstm_layers: int = 2, name: str = 'vrae'):
        super().__init__(
            name=name, input_size=input_size, encoding_size=encoding_size,
            condition_size=condition_size
        )
        self.beta = beta
        self.lstm_layers = lstm_layers

        self.lstm_encoder = tf.keras.layers.RNN(
            cell=[tf.keras.layers.LSTMCell(units=self.encoding_size) for _ in range(self.lstm_layers)],
            return_sequences=False, return_state=False)

        self.lstm_decoder = tf.keras.layers.RNN(
            cell=[tf.keras.layers.LSTMCell(units=self.encoding_size) for _ in range(self.lstm_layers)],
            return_sequences=True, return_state=True)

        self.gaussian = GaussianTransformation(input_size=self.encoding_size, output_size=self.encoding_size)
        self.mean = self.gaussian.mean
        self.stddev = self.gaussian.stddev

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

    def call(self, inputs, identifier=None, condition=(), return_encoding=False, series_dropout=0.,
             n_forecast=0) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:

        h_out = self.lstm_encoder(inputs)

        mean, stddev = self.gaussian(h_out)
        e_h = tf.random.normal(shape=tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)
        encoding_h = mean + stddev * e_h

        c_0 = tf.zeros(shape=tf.shape(encoding_h), dtype=tf.float32)
        initial_state = [[encoding_h, c_0]] * self.lstm_layers

        input_decoder = tf.concat((tf.expand_dims(encoding_h, axis=1), inputs[:, :-1, :]), axis=1)
        if series_dropout > 0.:
            input_decoder = tf.nn.dropout(input_decoder, rate=series_dropout)

        decoder_output = self.lstm_decoder(input_decoder, initial_state=initial_state)
        y = decoder_output[0]
        decoder_state = decoder_output[1:]

        if n_forecast > 0:
            y_forecast = self.lstm_loop(y_0=y[:, -1:, :], initial_state=decoder_state, n=n_forecast)
            y = tf.concat((y, y_forecast), axis=1)

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

    @tf.function
    @tensorflow_name_scoped
    def sample(self, n, condition=(), identifier=None):

        encoding_h = tf.random.normal(shape=(1, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32)
        c_0 = tf.zeros(shape=(1, self.encoding_size))
        initial_state = [[encoding_h, c_0]] * self.lstm_layers

        y = self.lstm_loop(y_0=tf.expand_dims(encoding_h, axis=0), initial_state=initial_state, n=n)

        return y

    @tf.function
    @tensorflow_name_scoped
    def lstm_loop(self, y_0, initial_state, n):
        y_i = y_0
        state_i = initial_state
        n = tf.cast(n, dtype=tf.int32)
        y = tf.TensorArray(dtype=tf.float32, size=n, clear_after_read=True)

        for i in tf.range(n):
            lstm_output = self.lstm_decoder(y_i, initial_state=state_i)
            y_i = lstm_output[0]
            state_i = lstm_output[1:]
            y = y.write(i, tf.squeeze(y_i, axis=0))

        z = tf.transpose(y.stack(), perm=(1, 0, 2))
        return z

    @property
    def regularization_losses(self):
        return [loss for layer in [self.gaussian] for loss in layer.regularization_losses]
