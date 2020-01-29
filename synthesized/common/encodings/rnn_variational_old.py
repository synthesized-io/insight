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
            output_size=self.encoding_size, return_state=True
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

        self.initial_input = self.add_module(
            module='dense', name='initial-input',
            input_size=(self.encoding_size + self.condition_size), output_size=self.encoding_size,
            batchnorm=False, activation='none'
        )

        self.initial_state = self.add_module(
            module='dense', name='initial-state',
            input_size=(self.encoding_size + self.condition_size), output_size=(2 * self.encoding_size),
            batchnorm=False, activation='none'
        )

        self.lstm_decoder = tf.keras.layers.LSTMCell(units=self.encoding_size)

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    def module_initialize(self):
        super().module_initialize()
        self.lstm_decoder.build(input_shape=(None, self.encoding_size))

    @tensorflow_name_scoped
    def encode(self, x, condition=(), encoding_plus_loss=False):
        print('x shape', x.shape)
        batch_size = tf.shape(input=x)[0]
        final_state = self.lstm_encoder.transform(x=x)
        final_state = tf.expand_dims(input=final_state, axis=0)

        mean = self.mean.transform(x=final_state)
        stddev = self.stddev.transform(x=final_state)

        encoding = tf.random.normal(
            shape=tf.shape(input=mean), mean=0.0, stddev=1.0, dtype=tf.float32
        )
        encoding = mean + stddev * encoding

        x = tf.concat(values=((encoding,) + tuple(condition)), axis=1)
        initial_input = self.initial_input.transform(x=x)
        initial_state = self.initial_state.transform(x=x)
        initial_state = [
            initial_state[:, :self.encoding_size], initial_state[:, self.encoding_size:]
        ]

        helper = RnnEncodingHelper(encoding_size=self.encoding_size, initial_input=initial_input)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.lstm_decoder, helper=helper, initial_state=initial_state
        )
        final_outputs, final_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, impute_finished=True, maximum_iterations=batch_size
        )
        x = final_outputs.sample_id[0]

        if encoding_plus_loss:
            encoding_loss = 0.5 * (tf.square(x=mean) + tf.square(x=stddev)) \
                - tf.math.log(x=tf.maximum(x=stddev, y=1e-6)) - 0.5
            encoding_loss = tf.reduce_mean(tf.reduce_sum(encoding_loss, axis=1), axis=0)

            if self.beta is not None:
                encoding_loss *= self.beta
            return x, encoding, encoding_loss
        else:
            return x

    @tensorflow_name_scoped
    def sample(self, n, condition=()):
        encoding = tf.random.normal(
            shape=(1, self.encoding_size), mean=0.0, stddev=1.0, dtype=tf.float32
        )

        x = tf.concat(values=((encoding,) + tuple(condition)), axis=1)
        initial_input = self.initial_input.transform(x=x)
        initial_state = self.initial_state.transform(x=x)
        initial_state = [
            initial_state[:, :self.encoding_size], initial_state[:, self.encoding_size:]
        ]

        helper = RnnEncodingHelper(encoding_size=self.encoding_size, initial_input=initial_input)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.lstm_decoder, helper=helper, initial_state=initial_state
        )
        final_outputs, final_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, impute_finished=True, maximum_iterations=tf.cast(x=n, dtype=tf.int32)
        )
        x = final_outputs.sample_id[0]

        return x


class RnnEncodingHelper(tf.contrib.seq2seq.CustomHelper):

    def __init__(self, encoding_size, initial_input):
        self.initial_input = initial_input

        super().__init__(
            initialize_fn=self.initialize_fn, sample_fn=self.sample_fn,
            next_inputs_fn=self.next_inputs_fn, sample_ids_shape=(encoding_size,),
            sample_ids_dtype=tf.float32
        )

    def initialize_fn(self):
        # finished, next_inputs
        return tf.zeros(shape=(1,), dtype=tf.bool), self.initial_input

    def sample_fn(self, time, outputs, state):
        # sample_ids
        return outputs

    def next_inputs_fn(self, time, outputs, state, sample_ids):
        # finished, next_inputs, next_state
        return tf.zeros(shape=(1,), dtype=tf.bool), sample_ids, state
