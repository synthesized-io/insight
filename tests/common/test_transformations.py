import numpy as np
import tensorflow as tf

from synthesized.common.transformations.dense import DenseTransformation
from synthesized.common.transformations.mlp import MlpTransformation
from synthesized.common.transformations.modulation import ModulationTransformation
from synthesized.common.transformations.residual import ResidualTransformation
from synthesized.common.transformations.resnet import ResnetTransformation


def _test_transformation(transformation, modulation=False):
    assert isinstance(transformation.specification(), dict)
    if modulation:
        transform_input = np.random.randn(4, 8).astype(np.float32)
        condition_input = np.random.randn(4, 6).astype(np.float32)
        transform_output = transformation(inputs=transform_input, condition=condition_input)
    else:
        transform_input = np.random.randn(4, 8).astype(np.float32)
        transform_output = transformation(inputs=transform_input)

    if modulation:
        transformed = transform_output
        assert transformed.shape == (4, 8)
    else:
        transformed = transform_output
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
