import tensorflow as tf

from ..module import Module


class Optimizer(Module):

    def __init__(self, name, algorithm='adam', learning_rate=3e-4, clip_gradients=None):
        super().__init__(name=name)
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.clip_gradients = clip_gradients

    def specification(self):
        spec = super().specification()
        spec.update(
            algorithm=self.algorithm, learning_rate=self.learning_rate,
            clip_gradients=self.clip_gradients
        )
        return spec

    def tf_initialize(self):
        super().tf_initialize()
        if self.algorithm == 'adam':
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8,
                use_locking=False, name='adam'
            )
        else:
            raise NotImplementedError

    def tf_optimize(self, loss, gradient_norms=False):
        grads_and_vars = self.optimizer.compute_gradients(
            loss=loss, var_list=None, aggregation_method=None, colocate_gradients_with_ops=False,
            grad_loss=None  # gate_gradients=GATE_OP
        )
        if gradient_norms:
            gradient_norms = dict()
            for grad, var in grads_and_vars:
                gradient_norms[var.name] = tf.norm(
                    tensor=grad, ord='euclidean', axis=None, keepdims=None
                )
        if self.clip_gradients is not None:
            for n in range(len(grads_and_vars)):
                grad, var = grads_and_vars[n]
                clipped_grad = tf.clip_by_value(
                    t=grad, clip_value_min=-self.clip_gradients, clip_value_max=self.clip_gradients
                )
                grads_and_vars[n] = (clipped_grad, var)
        optimized = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=None)
        if gradient_norms is False:
            return optimized
        else:
            return optimized, gradient_norms
