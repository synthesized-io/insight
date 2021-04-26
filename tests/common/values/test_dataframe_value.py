from typing import Dict, Sequence

import pytest
import tensorflow as tf

from synthesized.common.values import CategoricalValue, ContinuousValue, DataFrameValue
from synthesized.config import CategoricalConfig


class TestDataFrameValue:
    batch_size = 4
    num_categories = 10
    child_values = {
        "cont_value": ContinuousValue(name="cont_value"),
        "cat_value": CategoricalValue(name="cat_value", num_categories=num_categories,
                                      config=CategoricalConfig(moving_average=False)),
        "cont_value_nan": CategoricalValue(name="cont_value_nan", num_categories=2,
                                           config=CategoricalConfig(moving_average=False)),
    }

    @pytest.fixture(scope="class")
    def value(self):
        return DataFrameValue(name="value", values=self.child_values)

    @pytest.fixture(scope="class")
    def inputs(self):
        inputs: Dict[str, Sequence[tf.Tensor]] = {
            "cont_value": (tf.random.normal(shape=(self.batch_size,)),),
            "cat_value": (tf.random.uniform(shape=(self.batch_size,), maxval=self.num_categories, dtype=tf.int64),),
            "cont_value_nan": (tf.constant([0, 0, 1, 1]),),
        }
        return inputs

    @pytest.fixture(scope="class")
    def outputs(self, value):
        outputs = tf.concat((
            tf.random.normal(shape=(self.batch_size, value["cont_value"].learned_output_size())),
            tf.random.normal(shape=(self.batch_size, value["cat_value"].learned_output_size())),
            tf.random.normal(shape=(self.batch_size, value["cont_value_nan"].learned_output_size())),
        ), axis=-1)
        return outputs

    def test_unify_inputs(self, value, inputs):
        assert value.unify_inputs(inputs).shape[0] == self.batch_size

    def test_output_tensors(self, value, outputs):
        output_tensors: Dict[str, Sequence[tf.Tensor]] = value.output_tensors(outputs)
        for output_tensor in output_tensors.values():
            assert output_tensor[0].shape[0] == self.batch_size

    def test_loss(self, value, outputs, inputs):
        assert value.loss(outputs, inputs).shape == ()
