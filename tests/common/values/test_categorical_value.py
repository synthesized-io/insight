import pytest
import tensorflow as tf

from synthesized.common.values import CategoricalValue
from synthesized.config import CategoricalConfig


class TestCategoricalValue:
    batch_size = 4
    num_categories = 10

    @pytest.fixture(scope="class", params=[True, False])
    def value(self, request):
        value = CategoricalValue(name="value", num_categories=self.num_categories, config=CategoricalConfig(moving_average=request.param))
        return value

    @pytest.fixture(scope="class")
    def inputs(self):
        inputs = tf.random.uniform(shape=(self.batch_size,), maxval=self.num_categories + 1, minval=1, dtype=tf.int32)
        return (inputs,)

    @pytest.fixture(scope="class")
    def outputs(self, value):
        return tf.random.normal(shape=(self.batch_size, value.learned_output_size()))

    def test_unify_inputs(self, value, inputs):
        assert value.unify_inputs(inputs).shape[0] == self.batch_size

    def test_output_tensors(self, outputs, value):
        assert value.output_tensors(outputs)[0].shape == self.batch_size

    def test_loss(self, value, outputs, inputs):
        loss = value.loss(outputs, inputs)

        if not value.moving_average:
            # a moving average will change the loss for the same inputs so we can only apply this test without a moving average
            non_nan_inputs = inputs[0] != 0
            loss_wout_nans = value.loss(tf.boolean_mask(outputs, non_nan_inputs), (tf.boolean_mask(inputs[0], non_nan_inputs),))
            assert loss == loss_wout_nans

        assert loss.shape == ()
