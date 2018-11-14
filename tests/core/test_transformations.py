import numpy as np
import tensorflow as tf

from synthesized.core.transformations import DenseTransformation, MlpTransformation, \
    ResidualTransformation, ResnetTransformation


def _test_transformation(transformation):
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


def test_dense():
    transformation = DenseTransformation(
        name='dense', input_size=8, output_size=6, bias=True, batchnorm=True, activation='relu',
        weight_decay=0.1
    )
    _test_transformation(transformation=transformation)


def test_mlp():
    transformation = MlpTransformation(
        name='mlp', input_size=8, layer_sizes=(10, 6), weight_decay=0.1
    )
    _test_transformation(transformation=transformation)


def test_residual():
    transformation = ResidualTransformation(
        name='residual', input_size=8, output_size=6, depth=2, layer_type='dense', batchnorm=True,
        activation='relu', weight_decay=0.1
    )
    _test_transformation(transformation=transformation)


def test_resnet():
    transformation = ResnetTransformation(
        name='resnet', input_size=8, layer_sizes=(10, 6), weight_decay=0.1, depths=3
    )
    _test_transformation(transformation=transformation)
