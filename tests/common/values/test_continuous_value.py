import numpy as np
import pytest
import tensorflow as tf

from synthesized.common.values import ContinuousValue


class TestContinuousValue:
    batch_size = 4

    @pytest.fixture(scope="class")
    def value(self):
        value = ContinuousValue(name="value")
        return value

    @pytest.fixture(scope="class")
    def inputs(self):
        inputs = np.random.normal(size=(self.batch_size,))
        inputs[0] = np.nan
        inputs[2] = np.nan
        return (tf.constant(inputs, dtype=tf.float32),)

    @pytest.fixture(scope="class")
    def outputs(self, value):
        return tf.random.normal(shape=(self.batch_size, value.learned_output_size()))

    def test_unify_inputs(self, value, inputs):
        assert value.unify_inputs(inputs).shape[0] == self.batch_size

    def test_output_tensors(self, outputs, value):
        assert value.output_tensors(outputs)[0].shape == self.batch_size

    def test_loss(self, value, outputs, inputs):
        loss = value.loss(outputs, inputs)
        nan_mask = tf.math.logical_not(tf.math.is_nan(inputs[0]))

        non_nan_loss = value.loss(tf.boolean_mask(outputs, nan_mask), (tf.boolean_mask(inputs[0], nan_mask),))
        # as boolean mask drops rows of the batch we need to renormalize to compare losses
        non_nan_loss = non_nan_loss * (self.batch_size - 2) / self.batch_size

        assert loss == non_nan_loss

        assert loss.shape == ()
