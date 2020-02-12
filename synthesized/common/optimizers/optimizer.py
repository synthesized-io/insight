from typing import Callable, List
import tensorflow as tf

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
        learning_rate: tf.Tensor, decay_steps: int = None, decay_rate: float = None,
        initial_boost: int = 0,
        # Gradient clipping by global norm
        clip_gradients: float = None
    ):
        super().__init__(name=name)

        # Optimizer
        self.global_step = tf.summary.experimental.get_step()

        # Learning rate
        self._learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.initial_boost = initial_boost

        # Gradient clipping
        self.clip_gradients = clip_gradients
        if optimizer not in tf_optimizers:
            raise NotImplementedError
        if self.clip_gradients is not None:
            self.optimizer = tf_optimizers[optimizer](learning_rate=self.learning_rate, clipnorm=self.clip_gradients)
        else:
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
                lr = tf.constant(10.0, dtype=tf.float32) * self._learning_rate
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

        tf.summary.scalar(name='learning_rate', data=lr)

        return lr

    def module_initialize(self):
        super().module_initialize()

    def optimize(self, loss: Callable[..., tf.Tensor], variables: Callable[..., List[tf.Variable]]):
        """Optimize the given loss.

        Args:
            loss: Loss tensor.
            summarize_gradient_norms: Whether to add summaries for gradient norms.
            summarize_lr: Whether to add summaries for learning rate.

        Returns:
            The optimization operation.

        """
        # Trainable variables
        self.optimizer.minimize(loss=loss, var_list=variables)
        return
