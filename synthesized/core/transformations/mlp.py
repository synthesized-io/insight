import tensorflow as tf
from synthesized.core.transformations import Transformation


class MlpTransformation(Transformation):

    def __init__(self, name, input_size, output_size, layer_sizes=(), activation='relu'):
        super().__init__(name=name, input_size=input_size, output_size=output_size)
        self.layer_sizes = list(layer_sizes)
        self.activation = activation

    def _initialize(self):
        self.weights = list()
        self.biases = list()
        previous_size = self.input_size
        for n, layer_size in enumerate(self.layer_sizes):
            self.weights.append(tf.get_variable(
                name=('weight' + str(n)), shape=(previous_size, layer_size), dtype=tf.float32,
                initializer=None, regularizer=None, trainable=True, collections=None,
                caching_device=None, partitioner=None, validate_shape=True, use_resource=None,
                custom_getter=None, constraint=None
            ))
            self.biases.append(tf.get_variable(
                name=('bias' + str(n)), shape=(layer_size,), dtype=tf.float32,
                initializer=None, regularizer=None, trainable=True, collections=None,
                caching_device=None, partitioner=None, validate_shape=True, use_resource=None,
                custom_getter=None, constraint=None
            ))
            previous_size = layer_size
        self.weights.append(tf.get_variable(
            name=('weight' + str(len(self.layer_sizes))), shape=(previous_size, self.output_size),
            dtype=tf.float32, initializer=None, regularizer=None, trainable=True,
            collections=None, caching_device=None, partitioner=None, validate_shape=True,
            use_resource=None, custom_getter=None, constraint=None
        ))
        self.biases.append(tf.get_variable(
            name=('bias' + str(len(self.layer_sizes))), shape=(self.output_size,),
            dtype=tf.float32, initializer=None, regularizer=None, trainable=True,
            collections=None, caching_device=None, partitioner=None, validate_shape=True,
            use_resource=None, custom_getter=None, constraint=None
        ))

    def transform(self, x):
        for weight, bias in zip(self.weights, self.biases):
            x = tf.matmul(a=x, b=weight, name=None)
            x = tf.nn.bias_add(value=x, bias=bias, name=None)
            if self.activation == 'relu':
                x = tf.nn.relu(features=x, name=None)
            else:
                raise NotImplementedError
        return x
