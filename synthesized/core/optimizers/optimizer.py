import tensorflow as tf
from synthesized.core import Module


class Optimizer(Module):

    def __init__(self, name, algorithm='adam', learning_rate=3e-4, clip_gradients=1.0):
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

    def tf_optimize(self, loss):
        grads_and_vars = self.optimizer.compute_gradients(
            loss=loss, var_list=None, aggregation_method=None, colocate_gradients_with_ops=False,
            grad_loss=None  # gate_gradients=GATE_OP
        )
        if self.clip_gradients is not None:
            for n in range(len(grads_and_vars)):
                grad, var = grads_and_vars[n]
                clipped_grad = tf.clip_by_value(
                    t=grad, clip_value_min=-self.clip_gradients,
                    clip_value_max=self.clip_gradients, name=None
                )
                grads_and_vars[n] = (clipped_grad, var)
        optimized = self.optimizer.apply_gradients(
            grads_and_vars=grads_and_vars, global_step=None, name=None
        )
        return optimized
