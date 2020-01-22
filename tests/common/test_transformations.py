import numpy as np
import tensorflow as tf

from synthesized.common.transformations.dense import DenseTransformation
from synthesized.common.transformations.mlp import MlpTransformation
from synthesized.common.transformations.modulation import ModulationTransformation
from synthesized.common.transformations.residual import ResidualTransformation
from synthesized.common.transformations.resnet import ResnetTransformation


def _test_transformation(transformation, modulation=False):
    assert isinstance(transformation.specification(), dict)
    tf.compat.v1.reset_default_graph()
    transformation.initialize()
    if modulation:
        transform_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 8))
        condition_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 6))
        transform_output = transformation.transform(x=transform_input, condition=condition_input)
    else:
        transform_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 8))
        transform_output = transformation.transform(x=transform_input)
    initialize = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as session:
        session.run(fetches=initialize)
        if modulation:
            transformed = session.run(
                fetches=transform_output, feed_dict={
                    transform_input: np.random.randn(4, 8), condition_input: np.random.randn(4, 6)
                }
            )
            assert transformed.shape == (4, 8)
        else:
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
        name='mlp', input_size=8, layer_sizes=(10, 6), batchnorm=True,
        activation='relu', weight_decay=0.1
    )
    _test_transformation(transformation=transformation)


def test_modulation():
    transformation = ModulationTransformation(name='modulation', input_size=8, condition_size=6)
    _test_transformation(transformation=transformation, modulation=True)


def test_residual():
    transformation = ResidualTransformation(
        name='residual', input_size=8, output_size=6, depth=2, batchnorm=True,
        activation='relu', weight_decay=0.1
    )
    _test_transformation(transformation=transformation)


def test_resnet():
    transformation = ResnetTransformation(
        name='resnet', input_size=8, layer_sizes=(10, 6), depths=2,
        batchnorm=True, activation='relu', weight_decay=0.1
    )
    _test_transformation(transformation=transformation)
