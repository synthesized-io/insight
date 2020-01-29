import tensorflow as tf

from .transformation import Transformation
from ..module import tensorflow_name_scoped


class LstmTransformation(Transformation):

    def __init__(self, name, input_size, output_size, return_state=False, return_sequences=True,
                 return_state_and_x=False):

        # assert (return_state + return_sequences + return_state_and_x) == 1
        super().__init__(name=name, input_size=input_size, output_size=output_size)

        self.return_state = return_state
        self.return_sequences = return_sequences
        self.return_state_and_x = return_state_and_x

        if self.return_state:
            assert output_size % 2 == 0
            units = self.output_size // 2
        else:
            units = self.output_size

        # tf.keras.layers.CuDNNLSTM
        self.lstm = tf.keras.layers.LSTM(units=units, return_sequences=self.return_sequences, return_state=True)

    def specification(self):
        spec = super().specification()
        return spec

    def module_initialize(self):
        super().module_initialize()
        self.lstm.build(input_shape=(None, None, self.input_size))

    @tensorflow_name_scoped
    def transform(self, x, state=None):

        expand_squeeze = True if x.shape.ndims == 2 else False

        if expand_squeeze:
            x = tf.expand_dims(input=x, axis=0)

        if state is None:
            initial_state = None
        else:
            state = tf.cast(state, dtype=tf.float32)
            h0, c0 = tf.split(state, [self.input_size, self.input_size], axis=0)

            if expand_squeeze:
                h0 = tf.expand_dims(input=h0, axis=0)
                c0 = tf.expand_dims(input=c0, axis=0)

            initial_state = tf.map_fn(lambda s: s, [h0, c0])

        x, h, c = self.lstm(inputs=x, initial_state=initial_state)

        if self.return_state:
            state = tf.concat(values=(h, c), axis=1)

            if expand_squeeze:
                state = tf.squeeze(input=state, axis=0)
            return state

        if self.return_state_and_x:
            state = tf.concat(values=(h, c), axis=1)

            if expand_squeeze:
                x = tf.squeeze(input=x, axis=0)
                state = tf.squeeze(input=state, axis=0)
            return state, x

        else:
            if expand_squeeze:
                x = tf.squeeze(input=x, axis=0)
            return x
