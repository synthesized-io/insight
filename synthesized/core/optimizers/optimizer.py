import sys

import tensorflow as tf

from ..module import Module, tensorflow_name_scoped


class Optimizer(Module):

    def __init__(
        self, name, algorithm='adam', learning_rate=3e-4, decay_steps=None, decay_rate=None,
        clip_gradients=None
    ):
        super().__init__(name=name)
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.clip_gradients = clip_gradients

    def specification(self):
        spec = super().specification()
        spec.update(
            algorithm=self.algorithm, learning_rate=self.learning_rate,
            decay_steps=self.decay_steps, decay_rate=self.decay_rate,
            clip_gradients=self.clip_gradients
        )
        return spec

    def module_initialize(self):
        super().module_initialize()
        if self.decay_steps is None:
            assert self.decay_rate is None
            learning_rate = self.learning_rate
        else:
            assert self.decay_rate is not None
            learning_rate = tf.train.exponential_decay(
                learning_rate=self.learning_rate, global_step=Module.global_step,
                decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=False
            )
        if self.algorithm == 'adam':
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8
            )
        else:
            raise NotImplementedError

    @tensorflow_name_scoped
    def optimize(self, loss, gradient_norms=False):
        variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)

        with tf.control_dependencies(control_inputs=(loss,)):
            gradients = tf.gradients(ys=loss, xs=variables)

        # grads = tf.concat(values=[tf.reshape(tensor=grad, shape=(-1,)) for grad, _ in grads_and_vars], axis=0)
        # grad_mean, grad_variance = tf.nn.moments(x=grads, axes=(0,))

        assertions = [
            tf.debugging.assert_equal(x=tf.reduce_any(input_tensor=tf.math.is_nan(x=grad)), y=False)
            for grad in gradients
        ]

        # assertions.append(tf.print(grad_mean, grad_variance, self.optimizer._lr))

        with tf.control_dependencies(control_inputs=assertions):
            # gradients = [
            #     tf.where(condition=tf.math.is_nan(x=grad), x=tf.zeros_like(tensor=grad), y=grad)
            #     for grad in gradients if grad is not None
            # ]

            if self.clip_gradients is not None:
                gradients, grad_norm = tf.clip_by_global_norm(
                    t_list=gradients, clip_norm=self.clip_gradients
                )

            # for n in range(len(grads_and_vars)):
            #     grad, var = grads_and_vars[n]
            #     clipped_grad = tf.clip_by_value(
            #         t=grad, clip_value_min=-self.clip_gradients, clip_value_max=self.clip_gradients
            #     )
            #     grads_and_vars[n] = (clipped_grad, var)

        if gradient_norms:
            gradient_norms = dict()
            for grad, var in zip(gradients, variables):
                gradient_norms[var.name[:var.name.index(':')]] = tf.norm(
                    tensor=grad, ord='euclidean', axis=None, keepdims=None
                )
            if self.clip_gradients is not None:
                gradient_norms['all'] = grad_norm

        optimized = self.optimizer.apply_gradients(grads_and_vars=list(zip(gradients, variables)))
        # , global_step=Module.global_step  (incremented in synthesizer?!)

        if gradient_norms is False:
            return optimized
        else:
            return optimized, gradient_norms
