import tensorflow as tf
import tensorflow_probability as tfp

from .encoding import Encoding
from ..transformations import DenseTransformation, LstmTransformation
from ..module import tensorflow_name_scoped


class RnnVariationalEncoding(Encoding):

    def __init__(self, name, input_size, encoding_size, condition_size=0, beta=None):
        super().__init__(
            name=name, input_size=input_size, encoding_size=encoding_size,
            condition_size=condition_size
        )
        self.beta = beta

        self.lstm_encoder = LstmTransformation(
            name='lstm', input_size=self.input_size,
            output_size=self.encoding_size,
            return_state=False, return_sequences=False
        )

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

        self.lstm_decoder = LstmTransformation(
            name='lstm', input_size=self.encoding_size,
            output_size=self.input_size,
            return_state=False, return_sequences=True, return_state_and_x=True
        )

    def build(self, input_shape):
        self.lstm_encoder.build(input_shape)
        self.mean.build(self.encoding_size)
        self.stddev.build(self.encoding_size)
        self.lstm_decoder.build(self.encoding_size)
        self.built = True

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    def call(self, inputs, condition=()):
        hx = self.lstm_encoder(inputs)
        hx = tf.expand_dims(hx, axis=0)
        mean = self.mean(hx)
        stddev = self.stddev(hx)

        x = tf.random.normal(
            shape=tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32
        )
        x = mean + stddev * x

        encoding_x = tf.concat([x, inputs[:-1, :]], axis=0)
        _, y = self.lstm_decoder(encoding_x)

        kl_loss = 0.5 * (tf.square(x=mean) + tf.square(x=stddev)) - tf.math.log(x=tf.maximum(x=stddev, y=1e-6)) - 0.5
        kl_loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=kl_loss, axis=1), axis=0)
        kl_loss *= self.beta

        self.add_loss(kl_loss, inputs=inputs)
        tf.summary.scalar(name='kl-loss', data=kl_loss)
        tf.summary.histogram(name='mean', data=self.mean.output),
        tf.summary.histogram(name='stddev', data=self.stddev.output),
        tf.summary.histogram(name='posterior_distribution', data=x),
        tf.summary.image(
            name='latent_space_correlation',
            data=tf.abs(tf.reshape(tfp.stats.correlation(x), shape=(1, self.encoding_size, self.encoding_size, 1)))
        )

        return y

    @tensorflow_name_scoped
    def sample(self, n, condition=()):

        encoding = tf.random.normal(shape=(1, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32)
        y = self.lstm_loop(encoding, n)

        return y

    def lstm_loop(self, encoding, n, x=None):
        y = tf.Variable(tf.zeros(shape=(1, self.encoding_size)), dtype=tf.float32, trainable=False)

        state = tf.zeros(shape=(2 * self.encoding_size))

        iteration = tf.constant(0, dtype=tf.int64)

        def cond(_iteration, _encoding, _state, _y):
            return tf.less(_iteration, n)

        def body(_iteration, _encoding, _state, _y):
            if x is not None and n > 0:
                x_in = tf.expand_dims(x[n-1, :], axis=0)
                _state, _encoding = self.lstm_decoder.transform(x_in, state=_state)
            else:
                _state, _encoding = self.lstm_decoder.transform(_encoding, state=_state)
            _y = tf.concat([_y, _encoding], axis=0)
            return tf.add(_iteration, 1), _encoding, _state, _y

        iteration, encoding, state, y = tf.while_loop(
            cond, body, [iteration, encoding, state, y],
            shape_invariants=[iteration.get_shape(), encoding.get_shape(), state.get_shape(),
                              tf.TensorShape([None, self.encoding_size])]
        )

        y = y[1:]

        return y

    @property
    def regularization_losses(self):
        return [loss for layer in [self.mean, self.stddev] for loss in layer.regularization_losses]
