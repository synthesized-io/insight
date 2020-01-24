import tensorflow as tf

from ..module import Module, tensorflow_name_scoped


# TensorFlow optimizer implementations
tf_optimizers = dict(
    adam=tf.keras.optimizers.Adam
)


class Optimizer(tf.Module):
    """Optimizer."""

    def __init__(
        self, name: str,
        # Optimizer: "adam"
        optimizer: str,
        # Learning rate
        learning_rate: tf.Tensor, global_step: tf.Variable, decay_steps: int = None, decay_rate: float = None, initial_boost: int = 0,
        # Gradient clipping by global norm
        clip_gradients: float = None
    ):
        super().__init__(name=name)

        # Optimizer
        self.global_step = global_step

        # Learning rate
        self._learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.initial_boost = initial_boost

        # Gradient clipping
        self.clip_gradients = clip_gradients
        if optimizer not in tf_optimizers:
            raise NotImplementedError
        self.optimizer = tf_optimizers[optimizer](learning_rate=self.learning_rate)

    def specification(self):
        spec = dict(name=self._name)
        spec.update(
            optimizer=self.optimizer, learning_rate=self._learning_rate,
            decay_steps=self.decay_steps, decay_rate=self.decay_rate,
            initial_boost=self.initial_boost, clip_gradients=self.clip_gradients
        )
        return spec

    @tf.function
    def learning_rate(self) -> tf.Tensor:
        if self.initial_boost > 0:
            if self.global_step < self.initial_boost:
                lr = 10.0 * self._learning_rate
            else:
                lr = self._learning_rate
        else:
            lr = self._learning_rate

        if self.decay_steps is None:
            # Constant learning rate
            if self.decay_rate is not None:
                raise NotImplementedError

        else:
            # Exponentially decaying learning rate
            if self.decay_rate is None:
                raise NotImplementedError
            lr = lr * tf.math.pow(self.decay_rate, self.global_step/self.decay_steps)

        return lr

    def module_initialize(self):
        super().module_initialize()

    def optimize(self, loss, summarize_gradient_norms=False, summarize_lr=False):
        """Optimize the given loss.

        Args:
            loss: Loss tensor.
            summarize_gradient_norms: Whether to add summaries for gradient norms.
            summarize_lr: Whether to add summaries for learning rate.

        Returns:
            The optimization operation.

        """
        # Trainable variables
        variables = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        self.optimizer.minimize(loss=loss, var_list=variables)
        return

        # # Make sure loss operation is executed (for attached control flow)
        # with tf.control_dependencies(control_inputs=(loss,)):
        #
        #     # Loss gradients
        #     gradients = tf.gradients(ys=loss, xs=variables)
        #
        #     # Check that gradients are not NaN
        #     assertions = [
        #         tf.compat.v1.debugging.assert_equal(
        #             x=tf.reduce_any(input_tensor=tf.math.is_nan(x=grad)), y=False
        #         ) for grad in gradients
        #     ]
        #
        # with tf.control_dependencies(control_inputs=assertions):
        #     if self.clip_gradients is not None:
        #         # Global norm gradient clipping
        #         gradients, grad_norm = tf.clip_by_global_norm(
        #             t_list=gradients, clip_norm=self.clip_gradients
        #         )
        #
        # summaries = list()
        # if summarize_gradient_norms:
        #     # Summarize gradient norms
        #     for grad, var in zip(gradients, variables):
        #         summaries.append(tf.compat.v2.summary.scalar(
        #             name=(var.name[:var.name.index(':')] + '-gradient-norm'),
        #             data=tf.norm(tensor=grad, ord='euclidean'), step=self.global_step
        #         ))
        #     if self.clip_gradients is not None:
        #         # Add global gradient norm if clipping
        #         summaries.append(
        #             tf.compat.v2.summary.scalar(name='all-gradient-norm', data=grad_norm, step=self.global_step)
        #         )
        #
        # if summarize_lr:
        #     summaries.append(
        #         tf.compat.v2.summary.scalar(name='learning-rate', data=self.optimizer._lr, step=self.global_step)
        #     )
        #
        # # Make sure summary operations are executed
        # with tf.control_dependencies(control_inputs=summaries):
        #
        #     # Optimization step
        #     grads_and_vars = list(zip(gradients, variables))
        #     optimized = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # return optimized
