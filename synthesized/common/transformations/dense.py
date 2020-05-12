from typing import Optional, Dict, Any
from math import log

import tensorflow as tf

from .transformation import Transformation
from .batch_norm import BatchNorm
from ..util import get_initializer
from ..module import tensorflow_name_scoped


class DenseTransformation(Transformation):

    def __init__(
        self, name, input_size, output_size, bias=True, batch_norm=True, activation='relu'
    ):
        super(DenseTransformation, self).__init__(name=name, input_size=input_size, output_size=output_size)

        self.use_bias = bias
        self.activation = activation

        self.weight: Optional[tf.Tensor] = None
        self.bias: Optional[tf.Tensor] = None
        self.batch_norm: Optional[BatchNorm] = BatchNorm(input_size=output_size) if batch_norm else None

    def specification(self):
        spec = super().specification()
        spec.update(
            bias=self.bias, batch_norm=self.batch_norm, activation=self.activation
        )
        return spec

    @tensorflow_name_scoped
    def build(self, input_shape):
        shape = (self.input_size, self.output_size)
        initializer = get_initializer(initializer='glorot-normal')
        self.weight = self.add_weight(
            name='weight', shape=shape, dtype=tf.float32, initializer=initializer, trainable=True
        )
        self.add_regularization_weights(self.weight)

        shape = (self.output_size,)
        initializer = get_initializer(initializer='zeros')
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias', shape=shape, dtype=tf.float32, initializer=initializer, trainable=True
            )
            self.add_regularization_weights(self.bias)

        if self.batch_norm:
            self.batch_norm.build(input_shape=shape)

        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x = tf.matmul(
            a=inputs, b=self.weight, transpose_a=False, transpose_b=False, adjoint_a=False,
            adjoint_b=False, a_is_sparse=False, b_is_sparse=False
        )

        if self.bias is not None:
            x = tf.nn.bias_add(value=x, bias=self.bias)

        if self.batch_norm:
            x = self.batch_norm(x)

        if self.activation == 'none':
            pass
        elif self.activation == 'relu':
            x = tf.nn.relu(features=x)
        elif self.activation == 'leaky_relu':
            x = tf.nn.leaky_relu(features=x)
        elif self.activation == 'softplus':
            # division so that 0.0 is mapped to 1.0, as expected for instance when used for stddev
            x = tf.nn.softplus(features=x) / log(2.0)
        elif self.activation == 'tanh':
            x = tf.tanh(x=x)
        else:
            raise NotImplementedError

        self._output = x

        return x

    def get_variables(self) -> Dict[str, Any]:
        if not self.built:
            raise ValueError(self.name + " hasn't been built yet. ")
        assert self.weight is not None

        variables = super().get_variables()
        variables.update(
            weight=self.weight.numpy(),
            use_bias=self.use_bias,
            bias=self.bias.numpy() if self.bias is not None else None,
            activation=self.activation,
            batch_norm=self.batch_norm.get_variables() if self.batch_norm is not None else None
        )
        return variables

    def set_variables(self, variables: Dict[str, Any]):
        super().set_variables(variables)
        assert self.use_bias == variables['use_bias']

        if not self.built:
            self.build(self.input_size)
        assert self.weight is not None

        self.weight.assign(variables['weight'])
        if self.bias is not None:
            self.bias.assign(variables['bias'])

        self.activation = variables['activation']
        if self.batch_norm is not None:
            self.batch_norm.set_variables(variables['batch_norm'])
