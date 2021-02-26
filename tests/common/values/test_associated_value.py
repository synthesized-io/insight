import numpy as np
import pytest
import tensorflow as tf

from synthesized.common.values import AssociatedCategoricalValue, CategoricalValue
from synthesized.common.values.associated_categorical import (tf_joint_prob_tensor, tf_masked_probs,
                                                              unflatten_joint_sample)


class TestAssociatedCategoricalValue:
    batch_size = 4
    child_values = {
        "child_value_0": CategoricalValue("cv_0", num_categories=6),
        "child_value_1": CategoricalValue("cv_1", num_categories=3),
    }

    @pytest.fixture(scope="class")
    def value(self):
        return AssociatedCategoricalValue(values=self.child_values, binding_mask=np.ones((7, 4)))

    @pytest.fixture(scope="class")
    def inputs(self):
        return (
            tf.random.uniform(shape=(self.batch_size,), maxval=6, dtype=tf.int64),
            tf.random.uniform(shape=(self.batch_size,), maxval=3, dtype=tf.int64),
        )

    @pytest.fixture(scope="class")
    def outputs(self, value):
        return tf.concat((
            tf.random.normal(shape=(self.batch_size, value["child_value_0"].learned_output_size())),
            tf.random.normal(shape=(self.batch_size, value["child_value_1"].learned_output_size())),
        ), axis=-1)

    def test_unify_inputs(self, value, inputs):
        assert value.unify_inputs(inputs).shape[0] == self.batch_size

    def test_output_tensors(self, value, outputs):
        for tensor in value.output_tensors(outputs):
            assert tensor.shape[0] == self.batch_size

    def test_loss(self, value, outputs, inputs):
        loss = value.loss(outputs, inputs)
        assert loss.shape == ()

    def test_unflatten_sample(self, value):
        flattened_sample = tf.constant(
            [15, 10, 2, 0, 17], dtype=tf.int64
        )

        expected_output_tensors = (
            tf.constant([5, 3, 0, 0, 5], dtype=tf.int64),
            tf.constant([0, 1, 2, 0, 2], dtype=tf.int64)
        )

        output_tensors = unflatten_joint_sample(flattened_sample, list(value.values()))

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
