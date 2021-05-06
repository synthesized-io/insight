import numpy as np
import pytest
import tensorflow as tf

from synthesized.common.rules import Association
from synthesized.common.values import CategoricalValue, ContinuousValue, DataFrameValue
from synthesized.common.values.associated_categorical import (output_association_tensors, tf_joint_prob_tensor,
                                                              tf_masked_probs, unflatten_joint_sample)


class TestAssociation:
    batch_size = 4

    @pytest.fixture(scope="class")
    def df_value(self):
        return DataFrameValue("df_value", values={
            "value_0": CategoricalValue("value_0", num_categories=6),
            "value_1": CategoricalValue("value_1", num_categories=3),
            "value_2": ContinuousValue("value_2"),
            "value_2_nan": CategoricalValue("value_2_nan", num_categories=2),
        })

    @pytest.fixture(scope="class")
    def association(self):
        return Association(binding_mask=np.ones((6, 2)), associations=["value_0"], nan_associations=["value_2"])

    @pytest.fixture(scope="class")
    def ys_dict(self, df_value):
        return {
            value.name: tf.random.normal(shape=(self.batch_size, value.learned_output_size())) for value in df_value.values()
        }

    def test_output_association(self, association, ys_dict):
        output_tensors = output_association_tensors(association, ys_dict)

        assert "value_0" in output_tensors
        assert "value_1" not in output_tensors
        assert "value_2" not in output_tensors
        assert "value_2_nan" in output_tensors

    def test_output_association_raises_value_error(self, ys_dict):
        association_0 = Association(binding_mask=np.ones((2, 2)), associations=["value_0", "fake_name_1"])
        association_1 = Association(binding_mask=np.ones((2, 2)), associations=["value_0"], nan_associations=["fake_name_1"])

        with pytest.raises(ValueError):
            output_association_tensors(association_0, ys_dict)

        with pytest.raises(ValueError):
            output_association_tensors(association_1, ys_dict)

    def test_wrong_binding_mask_shape_raises_value_error(self, ys_dict):
        association = Association(binding_mask=np.ones((1, 1)), associations=["value_0", "value_1"])

        with pytest.raises(ValueError):
            output_association_tensors(association, ys_dict)


def test_unflatten_sample():
    flattened_sample = tf.constant(
        [15, 10, 2, 0, 17], dtype=tf.int64
    )

    expected_output_tensors = (
        tf.constant([5, 3, 0, 0, 5], dtype=tf.int64),
        tf.constant([0, 1, 2, 0, 2], dtype=tf.int64)
    )

    output_tensors = unflatten_joint_sample(flattened_sample, (6, 3))

    tf.assert_equal(output_tensors[0], expected_output_tensors[0])
    tf.assert_equal(output_tensors[1], expected_output_tensors[1])


def test_joint_probs():
    args = (
        tf.constant(np.array(
            [[0.9, 0.1, 0],
             [0.4, 0.4, 0.2]]
        )),
        tf.constant(np.array(
            [[0.5, 0.5],
             [0.2, 0.8]]
        ))
    )

    output = tf.constant(np.array(
        [[[0.45, 0.45],
          [0.05, 0.05],
          [0, 0]],
         [[0.08, 0.32],
          [0.08, 0.32],
          [0.04, 0.16]]]
    ))

    tf.debugging.assert_near(tf_joint_prob_tensor(*args), output)


def test_masked_probs():
    joint_probs = tf.constant(
        [[[0.4, 0.4],
          [0.05, 0.05],
          [0.05, 0.05]],
         [[0.08, 0.32],
          [0.08, 0.32],
          [0.04, 0.16]]]
    )

    mask = tf.constant(
        [[1, 1],
         [0, 1],
         [1, 1]], dtype=tf.float32
    )

    output_probs = tf.constant(
        [[[8 / 19, 8 / 19],
          [0, 1 / 19],
          [1 / 19, 1 / 19]],
         [[2 / 23, 8 / 23],
          [0, 8 / 23],
          [1 / 23, 4 / 23]]]
    )

    tf.debugging.assert_near(tf_masked_probs(joint_probs, mask), output_probs)
