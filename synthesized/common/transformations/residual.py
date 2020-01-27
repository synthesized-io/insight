import tensorflow as tf

from .transformation import Transformation
from .. import util
from .dense import DenseTransformation
from .linear import LinearTransformation


class ResidualTransformation(Transformation):

    def __init__(
        self, name, input_size, output_size, depth=2, batchnorm=True, activation='relu',
        weight_decay=0.0
    ):
        super().__init__(name=name, input_size=input_size, output_size=output_size)

        self.batchnorm = batchnorm
        self.activation = activation

        self.layers = list()
        for n in range(depth - 1):
            self.layers.append(DenseTransformation(
                name=('Dense_' + str(n)), input_size=input_size,
                output_size=input_size, batchnorm=batchnorm, activation=activation,
                weight_decay=weight_decay
            ))
        self.layers.append(DenseTransformation(
            name=('Dense_' + str(depth - 1)), input_size=input_size,
            output_size=output_size, weight_decay=weight_decay
        ))

        if input_size != output_size:
            self.identity_transformation = LinearTransformation(
                name='idtransform', input_size=input_size, output_size=output_size,
                weight_decay=weight_decay,
            )
        else:
            self.identity_transformation = None

    def specification(self):
        spec = super().specification()
        spec.update(
            layers=[layer.specification() for layer in self.layers],
            identity_transformation=(self.identity_transformation.specification()
                                     if self.identity_transformation else None)
        )
        return spec

    def build(self, input_shape):
        if self.batchnorm:
            shape = (self.output_size,)
            initializer = util.get_initializer(initializer='zeros')
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

    def call(self, inputs, **kwargs):
        residual = inputs
        for layer in self.layers:
            residual = layer(residual)

        if self.identity_transformation is not None:
            inputs = self.identity_transformation(inputs)

        x = inputs + residual

        if self.batchnorm:
            mean, variance = tf.nn.moments(x=x, axes=(0,), shift=None, keepdims=False)
            x = tf.nn.batch_normalization(
                x=x, mean=mean, variance=variance, offset=self.offset,
                scale=tf.nn.softplus(features=self.scale), variance_epsilon=1e-6
            )

        if self.activation == 'none':
            pass
        elif self.activation == 'relu':
            x = tf.nn.relu(features=x)
        elif self.activation == 'softplus':
            x = tf.nn.softplus(features=x)
        elif self.activation == 'tanh':
            x = tf.tanh(x=x)
        else:
            raise NotImplementedError

        # dropout
        return x
