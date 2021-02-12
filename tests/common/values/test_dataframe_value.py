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

    @pytest.fixture(scope="class")
    def loss_from_children(self, value, inputs, outputs):
        """ Drops the nan values manually and calculates the loss """
        cont_output_size = value["cont_value"].learned_output_size()
        cat_output_size = value["cat_value"].learned_output_size()
        nan_output_size = value["cont_value_nan"].learned_output_size()

        cont_outputs = outputs[:, :cont_output_size]
        cat_outputs = outputs[:, cont_output_size: cont_output_size + cat_output_size]
        nan_outputs = outputs[:, -nan_output_size:]

        dropped_cont_inputs = tf.boolean_mask(inputs["cont_value"][0], 1 - inputs["cont_value_nan"][0], axis=0)
        dropped_cont_outputs = tf.boolean_mask(cont_outputs, 1 - inputs["cont_value_nan"][0], axis=0)
        cont_loss = value["cont_value"].loss(dropped_cont_outputs, (dropped_cont_inputs,))

        cat_loss = value["cat_value"].loss(cat_outputs, inputs["cat_value"])
        nan_loss = value["cont_value_nan"].loss(nan_outputs, inputs["cont_value_nan"])

        return cont_loss + cat_loss + nan_loss

    def test_unify_inputs(self, value, inputs):
        assert value.unify_inputs(inputs).shape[0] == self.batch_size

    def test_output_tensors(self, value, outputs):
        output_tensors: Dict[str, Sequence[tf.Tensor]] = value.output_tensors(outputs)
        for output_tensor in output_tensors.values():
            assert output_tensor[0].shape[0] == self.batch_size

    def test_loss(self, value, outputs, inputs, loss_from_children):
        assert (value.loss(outputs, inputs) == loss_from_children).numpy()
