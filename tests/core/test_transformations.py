import unittest
import numpy as np
import tensorflow as tf
from synthesized.core.transformations import MlpTransformation


class TestTransformations(unittest.TestCase):

    def _test_transformation(self, transformation):
        tf.reset_default_graph()
        transformation.initialize()
        transform_input = tf.placeholder(dtype=tf.float32, shape=(None, 8))
        transform_output = transformation.transform(x=transform_input)
        initialize = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(fetches=initialize)
            transformed = session.run(
                fetches=transform_output, feed_dict={transform_input: np.random.randn(4, 8)}
            )
            assert transformed.shape == (4, 6)

    def test_mlp(self):
        transformation = MlpTransformation(
            name='mlp', input_size=8, output_size=6, layer_sizes=(10, 4), activation='relu'
        )
        self._test_transformation(transformation=transformation)
