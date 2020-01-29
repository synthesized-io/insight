import tensorflow as tf
from .encoding import Encoding
from ..module import tensorflow_name_scoped


class RnnVariationalEncoding(Encoding):

    def __init__(self, name, input_size, encoding_size, condition_size=0, beta=None):
        super().__init__(
            name=name, input_size=input_size, encoding_size=encoding_size,
            condition_size=condition_size
        )
        self.beta = beta

        self.lstm_encoder = self.add_module(
            module='lstm', name='lstm', input_size=self.input_size,
            output_size=self.encoding_size,
            return_state=False, return_sequences=False
        )

        self.mean = self.add_module(
            module='dense', name='mean',
            input_size=self.encoding_size, output_size=self.encoding_size, batchnorm=False,
            activation='none'
        )

        self.stddev = self.add_module(
            module='dense', name='stddev',
            input_size=self.encoding_size, output_size=self.encoding_size, batchnorm=False,
            activation='softplus'
        )

        self.lstm_decoder = self.add_module(
            module='lstm', name='lstm', input_size=self.encoding_size,
            output_size=self.input_size,
            return_state=False, return_sequences=True, return_state_and_x=True
        )

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    def module_initialize(self):
        super().module_initialize()

    @tensorflow_name_scoped
    def encode(self, x, condition=(), encoding_plus_loss=False):
        hx = self.lstm_encoder.transform(x=x)
        hx = tf.expand_dims(hx, axis=0)
        mean = self.mean.transform(x=hx)
        stddev = self.stddev.transform(x=hx)

        encoding = tf.random.normal(
            shape=tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32
        )
        encoding = mean + stddev * encoding

        # y = self.lstm_loop(encoding, 1000, x=x)

        encoding_x = tf.concat([encoding, x[:-1, :]], axis=0)
        _, y = self.lstm_decoder.transform(encoding_x)

        if encoding_plus_loss:
            encoding_loss = 0.5 * (tf.square(x=mean) + tf.square(x=stddev)) \
                            - tf.math.log(x=tf.maximum(x=stddev, y=1e-6)) - 0.5
            encoding_loss = tf.reduce_mean(tf.reduce_sum(encoding_loss, axis=1), axis=0)

            if self.beta is not None:
                encoding_loss *= self.beta
            return y, encoding, encoding_loss, mean, stddev
        else:
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
