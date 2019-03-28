import tensorflow as tf

from .transformation import Transformation
from .. import util
from ..module import tensorflow_name_scoped


class LstmTransformation(Transformation):

    def __init__(self, name, input_size, output_size, return_state=False):
        super().__init__(name=name, input_size=input_size, output_size=output_size)

        self.return_state = return_state

        if self.return_state:
            assert output_size % 2 == 0
            units = self.output_size // 2
        else:
            units = self.output_size

        # tf.keras.layers.CuDNNLSTM
        self.lstm = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True)

    def specification(self):
        spec = super().specification()
        return spec

    def module_initialize(self):
        super().module_initialize()
        self.lstm.build(input_shape=(None, None, self.input_size))

    #     initializer = util.get_initializer(initializer='random')
    #     self.state = tf.get_variable(
    #         name='state', shape=(2 * self.output_size,), dtype=tf.float32, initializer=initializer,
    #         regularizer=None, trainable=False, collections=None, caching_device=None,
    #         partitioner=None, validate_shape=True, use_resource=None, custom_getter=None
    #     )
    #
    # def tf_reset(self):
    #     initializer = util.get_initializer(initializer='normal')
    #     value = initializer(shape=(2 * self.output_size,))
    #     return self.state.assign(value=value, read_value=False)

    @tensorflow_name_scoped
    def transform(self, x, state=None):
        expand_squeeze = (x.shape.ndims == 2)

        if expand_squeeze:
            x = tf.expand_dims(input=x, axis=0)

        if state is None:
            x, h, c = self.lstm(inputs=x)
            # assignment = self.state.assign(value=next_state, read_value=False)
            # with tf.control_dependencies(control_inputs=(assignment,)):
            #     x += 0.0
        else:
            if state.shape.ndims == 1:
                state = tf.expand_dims(input=state, axis=0)
            state = (state[:, :state.shape.dims[1] // 2], state[:, state.shape.dims[1] // 2:])
            x, h, c = self.lstm(inputs=x, initial_state=state)

        state = tf.concat(values=(h, c), axis=1)

        if expand_squeeze:
            x = tf.squeeze(input=x, axis=0)
            state = tf.squeeze(input=state, axis=0)

        if self.return_state:
            return state
        else:
            return x
