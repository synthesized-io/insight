from typing import Dict, Any

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
            self.units = self.output_size // 2
        else:
            self.units = self.output_size

    def specification(self):
        spec = super().specification()
        return spec

    @tensorflow_name_scoped
    def build(self, input_shape):
        self.lstm = tf.keras.layers.LSTM(units=self.units, return_sequences=self.return_sequences, return_state=True)
        self.lstm.build(input_shape=(None, None, self.input_size))

    @tensorflow_name_scoped
    def call(self, inputs, ground_truth=None, state=None):

        expand_squeeze = True if inputs.shape.ndims == 2 else False

        if expand_squeeze:
            inputs = tf.expand_dims(input=inputs, axis=0)

        if ground_truth is not None:
            inputs[1:] = ground_truth[1:]

        if state is None:
            initial_state = None
        else:
            state = tf.cast(state, dtype=tf.float32)
            h0, c0 = tf.split(state, [self.input_size, self.input_size], axis=0)

            if expand_squeeze:
                h0 = tf.expand_dims(input=h0, axis=0)
                c0 = tf.expand_dims(input=c0, axis=0)

            initial_state = [h0, c0]

        outputs, h, c = self.lstm(inputs=inputs, initial_state=initial_state)

        if self.return_state:
            state = tf.concat(values=(h, c), axis=1)

            if expand_squeeze:
                state = tf.squeeze(input=state, axis=0)
            return state

        if self.return_state_and_x:
            state = tf.concat(values=(h, c), axis=1)

            if expand_squeeze:
                outputs = tf.squeeze(input=outputs, axis=0)
                state = tf.squeeze(input=state, axis=0)
            return state, outputs

        else:
            if expand_squeeze:
                outputs = tf.squeeze(input=outputs, axis=0)
            return outputs

    def get_variables(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_variables(self, variables: Dict[str, Any]):
        raise NotImplementedError
