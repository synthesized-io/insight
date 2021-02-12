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
        inputs = (tf.random.normal(shape=(self.batch_size,)),)
        return inputs

    @pytest.fixture(scope="class")
    def outputs(self, value):
        return tf.random.normal(shape=(self.batch_size, value.learned_output_size()))

    def test_unify_inputs(self, value, inputs):
        assert value.unify_inputs(inputs).shape[0] == self.batch_size

    def test_output_tensors(self, outputs, value):
        assert value.output_tensors(outputs)[0].shape == self.batch_size

    def test_loss(self, value, outputs, inputs):
        loss = value.loss(outputs, inputs)
        assert loss.shape == ()
