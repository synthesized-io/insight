import tensorflow as tf

from .transformation import Transformation
from .. import util


class ResidualTransformation(Transformation):

    def __init__(
        self, name, input_size, output_size, depth=2, layer_type='dense', batchnorm=True,
        activation='relu', **kwargs
    ):
        super().__init__(name=name, input_size=input_size, output_size=output_size)
        self.layer_type = layer_type
        self.depth = depth
        self.batchnorm = batchnorm
        self.activation = activation
        self.kwargs = kwargs

        from . import transformation_modules
        self.layers = list()
        if input_size != output_size:
            self.identity_transformation = self.add_module(
                module=self.layer_type, modules=transformation_modules, name='idtransform',
                input_size=input_size, output_size=output_size, batchnorm=False, activation='none',
                **self.kwargs
            )
        else:
            self.identity_transformation = None
        for n in range(self.depth - 1):
            self.layers.append(self.add_module(
                module=self.layer_type, modules=transformation_modules, name=('layer' + str(n)),
                input_size=input_size, output_size=input_size, batchnorm=self.batchnorm,
                activation=self.activation, **self.kwargs
            ))
        self.layers.append(self.add_module(
            module=self.layer_type, modules=transformation_modules,
            name=('layer' + str(self.depth - 1)), input_size=input_size, output_size=output_size,
            batchnorm=False, activation='none', **self.kwargs
        ))

    def specification(self):
        spec = super().specification()
        spec.update(
            layer_type=self.layer_type, num_layers=self.layers, batchnorm=self.batchnorm,
            activation=self.activation, kwargs=self.kwargs
        )
        return spec

    def tf_initialize(self):
        super().tf_initialize()
        if self.batchnorm:
            self.offset = tf.get_variable(
                name='offset', shape=(self.output_size,), dtype=tf.float32,
                initializer=util.initializers['zero'], regularizer=util.regularizers['l2'],
                trainable=True, collections=None, caching_device=None, partitioner=None,
                validate_shape=True, use_resource=None, custom_getter=None
            )
            self.scale = tf.get_variable(
                name='scale', shape=(self.output_size,), dtype=tf.float32,
                initializer=util.initializers['one'], regularizer=util.regularizers['l2'],
                trainable=True, collections=None, caching_device=None, partitioner=None,
                validate_shape=True, use_resource=None, custom_getter=None
            )

    def tf_transform(self, x):
        residual = x
        for layer in self.layers:
            residual = layer.transform(x=residual)
        if self.identity_transformation is not None:
            x = self.identity_transformation.transform(x=x)
        x = x + residual
        if self.batchnorm:
            mean, variance = tf.nn.moments(x=x, axes=(0,), shift=None, keep_dims=False)
            x = tf.nn.batch_normalization(
                x=x, mean=mean, variance=variance, offset=self.offset, scale=self.scale,
                variance_epsilon=1e-6
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
