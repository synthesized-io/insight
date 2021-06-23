from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

import tensorflow as tf
import tensorflow_privacy as privacy

from ...config import DifferentialPrivacyConfig


class Optimizer(tf.Module):
    """Optimizer."""

    # tensorflow optimizer implementations
    _tf_optimizers: Dict[str, Type[tf.optimizers.Optimizer]] = {
        'adam': tf.keras.optimizers.Adam,
        'adadelta': tf.keras.optimizers.Adadelta
    }

    def __init__(self, name: str, optimizer: str, learning_rate: Union[tf.Tensor, float], decay_steps: int = None,
                 decay_rate: Optional[float] = None, initial_boost: int = 0, clip_gradients: Optional[float] = None):
        super().__init__(name=name)

        # Optimizer
        self.global_step = tf.summary.experimental.get_step()

        # Learning rate
        self._learning_rate = tf.constant(learning_rate, dtype=tf.float32)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.initial_boost = initial_boost

        # Gradient clipping
        self.clip_gradients = clip_gradients
        self.optimizer_name = optimizer
        if optimizer not in self._tf_optimizers:
            raise ValueError(f"'optimizer' must be one of: {', '.join(self._tf_optimizers.keys())}. Not '{optimizer}'")
        self.optimizer = self.get_optimizer(self.optimizer_name)

    def get_optimizer(self, optimizer: str) -> tf.optimizers.Optimizer:
        return self._tf_optimizers[optimizer](
            name=self.optimizer_name, learning_rate=self.learning_rate, clipnorm=self.clip_gradients
        )

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
            lr = lr * tf.math.pow(self.decay_rate, self.global_step / self.decay_steps)

        tf.summary.scalar(name='learning_rate', data=lr)

        return lr

    def module_initialize(self):
        super().module_initialize()

    def optimize(self, loss: Callable[..., tf.Tensor], trainable_vars: Sequence[tf.Variable]):
        """Optimize the given loss.

        Args:
            loss: Callable which returns the total loss
            trainable_vars: List of trainable variables to optimize

        Returns:
            The optimization operation.

        """
        # Trainable variables

        with tf.name_scope("optimization"):
            self.optimizer.minimize(loss, trainable_vars)
        return

    def get_variables(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            clip_gradients=self.clip_gradients,
            optimizer_name=self.optimizer_name,
            learning_rate=self._learning_rate.numpy(),
            global_step=self.global_step.numpy(),
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            initial_boost=self.initial_boost,
        )

    def set_variables(self, variables: Dict[str, Any]):
        assert self.name == variables['name']
        assert self.clip_gradients == variables['clip_gradients']
        assert self.optimizer_name == variables['optimizer_name']
        assert self._learning_rate.numpy() == variables['learning_rate']

        self.global_step.assign(variables['global_step'])
        self.decay_steps = variables['decay_steps']
        self.decay_rate = variables['decay_rate']
        self.initial_boost = variables['initial_boost']


class DPOptimizer(Optimizer):
    """Optimizer that enables differential privacy."""

    # tensorflow_privacy optimizer implementations
    _tf_optimizers: Dict[str, Type[tf.optimizers.Optimizer]] = {
        'adam': privacy.DPKerasAdamOptimizer,
        'adagrad': privacy.DPKerasAdagradOptimizer,
        'sgd': privacy.DPKerasSGDOptimizer
    }

    def __init__(self, name: str, optimizer: str, learning_rate: Union[tf.Tensor, float], decay_steps: int = None,
                 decay_rate: float = None, initial_boost: int = 0, privacy_config: DifferentialPrivacyConfig = None):

        self.privacy_config = privacy_config or DifferentialPrivacyConfig()

        super().__init__(name=name, optimizer=optimizer, learning_rate=learning_rate, decay_steps=decay_steps,
                         decay_rate=decay_rate, initial_boost=initial_boost, clip_gradients=None)

    def get_optimizer(self, optimizer: str) -> tf.optimizers.Optimizer:
        return self._tf_optimizers[optimizer](
            name=self.optimizer_name, learning_rate=self.learning_rate,
            noise_multiplier=self.privacy_config.noise_multiplier,
            l2_norm_clip=self.privacy_config.l2_norm_clip,
            num_microbatches=self.privacy_config.num_microbatches
        )
