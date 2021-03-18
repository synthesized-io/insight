import pytest
import tensorflow as tf

from synthesized.common.values import DateValue


class TestDateValue:
    batch_size = 4

    @pytest.fixture(scope="class")
    def value(self):
        value = DateValue(name="value")
        return value

    @pytest.fixture(scope="class")
    def inputs(self):
        inputs = (
            tf.random.normal(shape=(self.batch_size,)),
            tf.random.uniform(shape=(self.batch_size,), minval=1, maxval=24, dtype=tf.int64),  # hour
            tf.random.uniform(shape=(self.batch_size,), minval=1, maxval=7, dtype=tf.int64),  # dow
            tf.random.uniform(shape=(self.batch_size,), minval=1, maxval=31, dtype=tf.int64),  # day
            tf.random.uniform(shape=(self.batch_size,), minval=1, maxval=12, dtype=tf.int64),  # month
        )
        return inputs

    @pytest.fixture(scope="class")
    def outputs(self, value):
        return tf.random.normal(shape=(self.batch_size, value.learned_output_size()))

    def test_unify_inputs(self, value, inputs):
        assert value.unify_inputs(inputs).shape[0] == self.batch_size

    def test_output_tensors(self, value, outputs):
        assert value.output_tensors(outputs)[0].shape == self.batch_size

    def test_loss(self, value, outputs, inputs):
        loss = value.loss(outputs, inputs)
        assert loss.shape == ()
