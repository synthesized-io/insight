import numpy as np
import tensorflow as tf

from .functional import Functional
from ..module import tensorflow_name_scoped


class CorrelationMatrixFunctional(Functional):

    def __init__(self, correlation_matrix, values='*', name=None):
        correlation_matrix = np.asarray(correlation_matrix)
        assert correlation_matrix.ndim == 2
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
        assert all(correlation_matrix[x, x] == 1.0 for x in range(correlation_matrix.shape[0]))
        assert all(
            correlation_matrix[x, y] == correlation_matrix[y, x]
            for x in range(correlation_matrix.shape[0]) for y in range(x)
        )
        assert values == '*' or len(set(values)) == len(values)

        super().__init__(values=values, name=name)

        self.correlation_matrix = correlation_matrix

    def specification(self):
        spec = super().specification()
        spec.update(correlation_matrix=self.correlation_matrix)
        return spec

    @tensorflow_name_scoped
    def loss(self, *samples_args):
        samples = tf.stack(values=samples_args, axis=0)
        means, variances = tf.nn.moments(x=samples, axes=1)
        means = tf.stop_gradient(input=means)
        means = tf.expand_dims(input=means, axis=1)
        outputs_x = tf.expand_dims(input=samples, axis=1)
        outputs_y = tf.expand_dims(input=samples, axis=0)
        means_x = tf.expand_dims(input=means, axis=1)
        means_y = tf.expand_dims(input=means, axis=0)
        covariance_matrix = tf.reduce_mean(
            input_tensor=((outputs_x - means_x) * (outputs_y - means_y)), axis=2
        )
        variances_x = tf.expand_dims(input=variances, axis=1)
        variances_y = tf.expand_dims(input=variances, axis=0)
        correlation_matrix = covariance_matrix / tf.sqrt(x=variances_x) / tf.sqrt(x=variances_y)
        loss = tf.squared_difference(x=correlation_matrix, y=self.correlation_matrix)
        # loss = tf.Print(loss, (correlation_matrix,), summarize=9)
        loss = tf.reduce_mean(input_tensor=loss, axis=(0, 1))
        return loss

    def check_distance(self, *samples_args):
        return abs(np.mean(
            np.cov(samples_args) / np.expand_dims(np.std(samples_args, 1), 1)
            / np.expand_dims(np.std(samples_args, 1), 2) - self.correlation_matrix
        ))
