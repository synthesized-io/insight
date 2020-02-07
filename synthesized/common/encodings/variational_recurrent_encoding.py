from typing import Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp

from .encoding import Encoding
from ..transformations import DenseTransformation
from ..module import tensorflow_name_scoped


class VariationalRecurrentEncoding(Encoding):

    def __init__(self, name, input_size, encoding_size, condition_size=0, beta=None, dropout=0.):
        super().__init__(
            name=name, input_size=input_size, encoding_size=encoding_size,
            condition_size=condition_size
        )
        self.beta = beta
        self.dropout = dropout

        self.lstm_encoder = tf.keras.layers.LSTM(units=encoding_size, return_state=True)

        self.mean = DenseTransformation(
            name='mean',
            input_size=self.encoding_size, output_size=self.encoding_size, batchnorm=False,
            activation='none'
        )

        self.stddev = DenseTransformation(
            name='stddev',
            input_size=self.encoding_size, output_size=self.encoding_size, batchnorm=False,
            activation='softplus'
        )

        self.lstm_decoder = tf.keras.layers.LSTM(units=encoding_size, return_sequences=True, return_state=True,
                                                 dropout=dropout)

    def build(self, input_shape):
        self.lstm_encoder.build(input_shape=(None, None, self.input_size))
        self.mean.build(self.encoding_size)
        self.stddev.build(self.encoding_size)
        self.lstm_decoder.build(input_shape=(None, None, self.encoding_size))
        self.built = True

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    def call(self, inputs, identifier=None, condition=(), return_encoding=False)\
            -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:

        expand_squeeze = True if inputs.shape.ndims == 2 else False
        if expand_squeeze:
            inputs = tf.expand_dims(input=inputs, axis=0)

        _, h_out, _ = self.lstm_encoder(inputs)
        mean = self.mean(h_out)
        stddev = self.stddev(h_out)

        e_h = tf.random.normal(shape=tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)
        encoding_h = mean + stddev * e_h

        encoding_c = tf.zeros(shape=tf.shape(mean), dtype=tf.float32)
        encoding = [encoding_h, encoding_c]

        y, _, _ = self.lstm_decoder(inputs, initial_state=encoding)

        kl_loss = 0.5 * (tf.square(x=mean) + tf.square(x=stddev)) - tf.math.log(x=tf.maximum(x=stddev, y=1e-6)) - 0.5
        kl_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=kl_loss, axis=1), axis=0)
        kl_loss *= self.beta

        self.add_loss(kl_loss, inputs=inputs)
        tf.summary.scalar(name='kl-loss', data=kl_loss)
        tf.summary.histogram(name='mean', data=self.mean.output),
        tf.summary.histogram(name='stddev', data=self.stddev.output),
        tf.summary.histogram(name='posterior_distribution', data=encoding_h),
        tf.summary.image(
            name='latent_space_correlation',
            data=tf.abs(tf.reshape(tfp.stats.correlation(encoding_h),
                                   shape=(1, self.encoding_size, self.encoding_size, 1)))
        )

        if expand_squeeze:
            y = tf.squeeze(input=y, axis=0)
            encoding_h = tf.squeeze(input=encoding_h, axis=0)

        if return_encoding:
            return y, encoding_h
        else:
            return y

    @tf.function
    @tensorflow_name_scoped
    def sample(self, n, condition=()):

        n = tf.cast(n, dtype=tf.int32)
        h_i = tf.random.normal(shape=(1, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32)
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
        return [loss for layer in [self.mean, self.stddev] for loss in layer.regularization_losses]
