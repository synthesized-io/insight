from typing import Dict, Any, Optional

import tensorflow as tf

from .dense import DenseTransformation
from .linear import LinearTransformation
from .transformation import Transformation
from ..util import get_initializer, check_params_version
from ..module import tensorflow_name_scoped


class ResidualTransformation(Transformation):

    def __init__(self, name, input_size, output_size, depth=2, batch_norm=True, activation='relu'):
        super().__init__(name=name, input_size=input_size, output_size=output_size)
        self.params_version = '0.0'

        self.depth = depth
        self.batch_norm = batch_norm
        self.activation = activation

        self.layers = list()
        for n in range(depth - 1):
            self.layers.append(DenseTransformation(
                name=('Dense_' + str(n)), input_size=input_size,
                output_size=input_size, batch_norm=batch_norm, activation=activation
            ))
        self.layers.append(DenseTransformation(
            name=('Dense_' + str(depth - 1)), input_size=input_size,
            output_size=output_size
        ))

        if input_size != output_size:
            self.identity_transformation = LinearTransformation(
                name='idtransform', input_size=input_size, output_size=output_size
            )
        else:
            self.identity_transformation = None

        self.offset: Optional[tf.Variable] = None
        self.scale: Optional[tf.Variable] = None

    def specification(self):
        spec = super().specification()
        spec.update(
            layers=[layer.specification() for layer in self.layers],
            identity_transformation=(self.identity_transformation.specification()
                                     if self.identity_transformation else None)
        )
        return spec

    @tensorflow_name_scoped
    def build(self, input_shape):
        if self.batch_norm:
            shape = (self.output_size,)
            initializer = get_initializer(initializer='zeros')
            self.offset = self.add_weight(
                name='offset', shape=shape, dtype=tf.float32, initializer=initializer,
                trainable=True,
            )
            self.scale = self.add_weight(
                name='scale', shape=shape, dtype=tf.float32, initializer=initializer,
                trainable=True,
            )

        if self.identity_transformation is not None:
            self.identity_transformation.build(input_shape)

        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

        self.built = True

    @tf.function(input_signature=[tf.TensorSpec(name='inputs', shape=None, dtype=tf.float32)])
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        residual = inputs
        for layer in self.layers:
            residual = layer(residual)

        if self.identity_transformation is not None:
            inputs = self.identity_transformation(inputs)

        x = inputs + residual

        if self.batch_norm:
            mean, variance = tf.nn.moments(x=x, axes=(0,), shift=None, keepdims=False)
            x = tf.nn.batch_normalization(
                x=x, mean=mean, variance=variance, offset=self.offset,
                scale=tf.nn.softplus(features=self.scale), variance_epsilon=1e-6
            )

        if self.activation == 'none':
            pass
        elif self.activation == 'relu':
            x = tf.nn.relu(features=x)
        elif self.activation == 'leaky_relu':
            x = tf.nn.leaky_relu(features=x)
        elif self.activation == 'softplus':
            x = tf.nn.softplus(features=x)
        elif self.activation == 'tanh':
            x = tf.tanh(x=x)
        else:
            raise NotImplementedError

        # dropout
        return x

    @property
    def regularization_losses(self):
        losses = [loss for layer in self.layers for loss in layer.regularization_losses]
        if self.identity_transformation is not None:
            losses.extend(self.identity_transformation.regularization_losses)

        return losses

    def get_variables(self) -> Dict[str, Any]:
        if not self.built:
            raise ValueError(self.name + " hasn't been built yet. ")

        variables = super().get_variables()
        variables.update(
            params_version=self.params_version,
            depth=self.depth,
            batch_norm=self.batch_norm,
            activation=self.activation
        )
        for i, layer in enumerate(self.layers):
            variables['layer_{}'.format(i)] = layer.get_variables()

        if self.identity_transformation is not None:
            variables['identity_transformation'] = self.identity_transformation.get_variables()

        if self.offset is not None and self.scale is not None:
            variables['offset'] = self.offset.numpy()
            variables['scale'] = self.scale.numpy()

        return variables

    def set_variables(self, variables: Dict[str, Any]):
        check_params_version(self.params_version, variables['params_version'])

        super().set_variables(variables)

        assert self.depth == variables['depth']
        assert self.batch_norm == variables['batch_norm']
        assert self.activation == variables['activation']

        if not self.built:
            self.build(self.input_size)

        for i, layer in enumerate(self.layers):
            layer.set_variables(variables['layer_{}'.format(i)])

        if self.identity_transformation is not None:
            self.identity_transformation.set_variables(variables['identity_transformation'])

        if self.offset is not None and self.scale is not None:
            self.offset.assign(variables['offset'])
            self.scale.assign(variables['scale'])
