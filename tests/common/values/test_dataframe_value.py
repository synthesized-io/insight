from typing import Dict, Sequence

import numpy as np
import pytest
import tensorflow as tf

from synthesized.common.rules import Association
from synthesized.common.values import CategoricalValue, ContinuousValue, DataFrameValue
from synthesized.config import CategoricalConfig


class TestDataFrameValue:
    batch_size = 4
    num_categories = 5
    child_values = {
        "cont_value": ContinuousValue(name="cont_value"),
        "cat_value_1": CategoricalValue(name="cat_value_1", num_categories=num_categories,
                                        config=CategoricalConfig(moving_average=False)),
        "cat_value_2": CategoricalValue(name="cat_value_2", num_categories=num_categories,
                                        config=CategoricalConfig(moving_average=False)),
        "cat_value_3": CategoricalValue(name="cat_value_3", num_categories=num_categories,
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
            "cat_value_1": (tf.random.uniform(shape=(self.batch_size,), maxval=self.num_categories, dtype=tf.int64),),
            "cat_value_2": (tf.random.uniform(shape=(self.batch_size,), maxval=self.num_categories, dtype=tf.int64),),
            "cat_value_3": (tf.random.uniform(shape=(self.batch_size,), maxval=self.num_categories, dtype=tf.int64),),
            "cont_value_nan": (tf.constant([0, 0, 1, 1]),),
        }
        return inputs

    @pytest.fixture(scope="class")
    def outputs(self, value):
        outputs = tf.concat((
            tf.random.normal(shape=(self.batch_size, value["cont_value"].learned_output_size())),
            tf.random.normal(shape=(self.batch_size, value["cat_value_1"].learned_output_size())),
            tf.random.normal(shape=(self.batch_size, value["cat_value_2"].learned_output_size())),
            tf.random.normal(shape=(self.batch_size, value["cat_value_3"].learned_output_size())),
            tf.random.normal(shape=(self.batch_size, value["cont_value_nan"].learned_output_size())),
        ), axis=-1)
        return outputs

    def test_unify_inputs(self, value, inputs):
        assert value.unify_inputs(inputs).shape[0] == self.batch_size

    def test_output_tensors(self, value, outputs):
        output_tensors: Dict[str, Sequence[tf.Tensor]] = value.output_tensors(outputs)
        for output_tensor in output_tensors.values():
            assert output_tensor[0].shape[0] == self.batch_size

    def test_output_tensors_w_association(self, value, outputs):
        association_1 = Association(binding_mask=np.array([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]),
                                    associations=["cat_value_1"], nan_associations=["cont_value"])

        association_2 = Association(binding_mask=np.ones((self.num_categories, self.num_categories)),
                                    associations=["cat_value_2", "cat_value_3"])

        output_tensors = value.output_tensors(outputs, association_rules=[association_1, association_2])
        for output_tensor in output_tensors.values():
            assert output_tensor[0].shape[0] == self.batch_size

        assert tf.math.reduce_all(output_tensors["cat_value_1"][0] == 1)
        assert tf.math.reduce_all(output_tensors["cont_value_nan"][0] == 1)

    def test_loss(self, value, outputs, inputs):
        assert value.loss(outputs, inputs).shape == ()
