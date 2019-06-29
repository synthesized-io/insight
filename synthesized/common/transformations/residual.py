import tensorflow as tf

from .transformation import Transformation
from ..module import tensorflow_name_scoped
from .. import util


class ResidualTransformation(Transformation):

    def __init__(
        self, name, input_size, output_size, depth=1, batchnorm=True, activation='relu',
        weight_decay=0.0
    ):
        super().__init__(name=name, input_size=input_size, output_size=output_size)

        self.batchnorm = batchnorm
        self.activation = activation

        self.layers = list()
        for n in range(depth - 1):
            self.layers.append(self.add_module(
                module='dense', name=('layer' + str(n)), input_size=input_size,
                output_size=input_size, batchnorm=batchnorm, activation=activation,
                weight_decay=weight_decay
            ))
        self.layers.append(self.add_module(
            module='linear', name=('layer' + str(depth - 1)), input_size=input_size,
            output_size=output_size, weight_decay=weight_decay
        ))

        if input_size != output_size:
            self.identity_transformation = self.add_module(
                module='linear', name='idtransform', input_size=input_size, output_size=output_size,
                weight_decay=weight_decay,
            )
        else:
            self.identity_transformation = None

    def specification(self):
        spec = super().specification()
        spec.update(
            layers=[layer.specification() for layer in self.layers],
            identity_transformation=self.identity_transformation.specification()
        )
        return spec

    def module_initialize(self):
        super().module_initialize()

        if self.batchnorm:
            shape = (self.output_size,)
            initializer = util.get_initializer(initializer='zeros')
            self.offset = tf.get_variable(
                name='offset', shape=shape, dtype=tf.float32, initializer=initializer,
                trainable=True,
            )
            self.scale = tf.get_variable(
                name='scale', shape=shape, dtype=tf.float32, initializer=initializer,
                trainable=True,
            )

    @tensorflow_name_scoped
    def transform(self, x):
        residual = x
        for layer in self.layers:
            residual = layer.transform(x=residual)

        if self.identity_transformation is not None:
            x = self.identity_transformation.transform(x=x)
        x = x + residual

        if self.batchnorm:
            mean, variance = tf.nn.moments(x=x, axes=(0,), shift=None, keep_dims=False)
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
