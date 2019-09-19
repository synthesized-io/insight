import tensorflow as tf

from ..module import Module, tensorflow_name_scoped


# TensorFlow optimizer implementations
tf_optimizers = dict(
    adam=tf.compat.v1.train.AdamOptimizer
)


class Optimizer(Module):
    """Optimizer."""

    def __init__(
        self, name: str,
        # Optimizer: "adam"
        optimizer: str,
        # Learning rate
        parent: Module,
        learning_rate: float, decay_steps: int = None, decay_rate: float = None,
        initial_boost: bool = False,
        # Gradient clipping by global norm
        clip_gradients: float = None
    ):
        super().__init__(name=name)

        # Optimizer
        if optimizer not in tf_optimizers:
            raise NotImplementedError
        self.optimizer = optimizer
        self.parent = parent

        # Learning rate
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.initial_boost = initial_boost

        # Gradient clipping
        self.clip_gradients = clip_gradients

    def specification(self):
        spec = super().specification()
        spec.update(
            optimizer=self.optimizer, learning_rate=self.learning_rate,
            decay_steps=self.decay_steps, decay_rate=self.decay_rate,
            initial_boost=self.initial_boost, clip_gradients=self.clip_gradients
        )
        return spec

    def module_initialize(self):
        super().module_initialize()

        # Learning rate
        if self.decay_steps is None:
            # Constant learning rate
            if self.decay_rate is not None:
                raise NotImplementedError
            learning_rate = self.learning_rate

        else:
            # Exponentially decaying learning rate
            if self.decay_rate is None:
                raise NotImplementedError
            learning_rate = tf.train.exponential_decay(
                learning_rate=self.learning_rate, global_step=self.parent.global_step,
                decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=False
            )

        if self.initial_boost:
            learning_rate = tf.where(
                condition=tf.math.less(x=self.parent.global_step, y=10),
                x=(5.0 * learning_rate), y=learning_rate
            )

        # TensorFlow optimizer
        self.optimizer = tf_optimizers[self.optimizer](learning_rate=learning_rate)

    @tensorflow_name_scoped
    def optimize(self, loss, summarize_gradient_norms=False):
        """Optimize the given loss.

        Args:
            loss: Loss tensor.
            summarize_gradient_norms: Whether to add summaries for gradient norms.

        Returns:
            The optimization operation.

        """
        # Trainable variables
        variables = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

        # Make sure loss operation is executed (for attached control flow)
        with tf.control_dependencies(control_inputs=(loss,)):

            # Loss gradients
            gradients = tf.gradients(ys=loss, xs=variables)

            # Check that gradients are not NaN
            assertions = [
                tf.compat.v1.debugging.assert_equal(
                    x=tf.reduce_any(input_tensor=tf.math.is_nan(x=grad)), y=False
                ) for grad in gradients
            ]

        with tf.control_dependencies(control_inputs=assertions):
            if self.clip_gradients is not None:
                # Global norm gradient clipping
                gradients, grad_norm = tf.clip_by_global_norm(
                    t_list=gradients, clip_norm=self.clip_gradients
                )

        summaries = list()
        if summarize_gradient_norms:
            # Summarize gradient norms
            for grad, var in zip(gradients, variables):
                summaries.append(tf.contrib.summary.scalar(
                    name=(var.name[:var.name.index(':')] + '-gradient-norm'),
                    tensor=tf.norm(tensor=grad, ord='euclidean')
                ))
            if self.clip_gradients is not None:
                # Add global gradient norm if clipping
                summaries.append(
                    tf.contrib.summary.scalar(name='all-gradient-norm', tensor=grad_norm)
                )

        # Make sure summary operations are executed
        with tf.control_dependencies(control_inputs=summaries):

            # Optimization step
            grads_and_vars = list(zip(gradients, variables))
            optimized = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        return optimized
