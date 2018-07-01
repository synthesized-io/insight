import unittest
import numpy as np
import tensorflow as tf
from synthesized.core.values import CategoricalValue, ContinuousValue


class TestValues(unittest.TestCase):

    def _test_value(self, value, x, equal=True):
        tf.reset_default_graph()
        value.initialize()
        input_tensor_output = value.input_tensor()
        output_tensor_output = value.output_tensor(x=input_tensor_output)
        loss_output = value.loss(x=input_tensor_output)
        initialize = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(fetches=initialize)
            input_tensor = session.run(
                fetches=input_tensor_output, feed_dict={value.placeholder: x}
            )
            assert input_tensor.shape == (4, value.size())
            output_tensor = session.run(
                fetches=output_tensor_output, feed_dict={value.placeholder: x}
            )
            assert output_tensor.shape == x.shape == (4,)
            assert not equal or np.allclose(output_tensor, x)
            loss = session.run(
                fetches=loss_output, feed_dict={value.placeholder: x}
            )
            assert loss.shape == () and loss >= 0.0

    def test_categorical(self):
        value = CategoricalValue(name='categorical', num_categories=8)
        self._test_value(value=value, x=np.asarray([0, 3, 5, 7]))

    def test_continuous(self):
        value = ContinuousValue(name='continuous', positive=False)
        self._test_value(value=value, x=np.asarray([-10.0, -0.1, 0.1, 10.0]))

    def test_continuous_positive(self):
        value = ContinuousValue(name='continuous', positive=True)
        self._test_value(value=value, x=np.asarray([0.1, 1.0, 10.0, 100.0]), equal=False)
