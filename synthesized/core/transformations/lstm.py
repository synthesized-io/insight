import tensorflow as tf

from .transformation import Transformation
from .. import util


class LstmTransformation(Transformation):

    def __init__(self, name, input_size, output_size):
        super().__init__(name=name, input_size=input_size, output_size=output_size)

    def specification(self):
        spec = super().specification()
        return spec

    def tf_initialize(self):
        super().tf_initialize()

        # tf.keras.layers.CuDNNLSTM
        self.lstm = tf.keras.layers.LSTM(
            units=self.output_size, return_sequences=True, return_state=True
        )
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

    def tf_transform(self, x, state=None):
        expand_squeeze = (x.shape.ndims == 2)

        if expand_squeeze:
            x = tf.expand_dims(input=x, axis=0)

        if state is None:
            x, _, next_state = self.lstm(inputs=x)
            # assignment = self.state.assign(value=next_state, read_value=False)
            # with tf.control_dependencies(control_inputs=(assignment,)):
            #     x += 0.0
        else:
            if state.shape.ndims == 1:
                state = tf.expand_dims(input=state, axis=0)
            state = (state[:, :state.shape.dims[1] // 2], state[:, state.shape.dims[1] // 2:])
            x, _, state = self.lstm(inputs=x, initial_state=state)

        if expand_squeeze:
            x = tf.squeeze(input=x, axis=0)

        # if state is None:
        #     return x
        # else:
        #     return x, state
        return x
