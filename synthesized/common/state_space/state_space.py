from typing import Tuple, Dict

import tensorflow as tf

from ..optimizers import Optimizer
from ..values import ValueFactory


class BaseStateSpaceModel(tf.Module):
    """The base class for state space models"""
    def __init__(self, df, input_size, capacity, latent_size, name='state_space_model'):
        super(BaseStateSpaceModel, self).__init__(name=name)
        self.input_size = input_size
        self.capacity = capacity
        self.latent_size = latent_size
        self._trainable_variables = None

        self.value_factory = ValueFactory(df, capacity=capacity)
        self.optimizer = Optimizer(name='optimizer', optimizer='adam', clip_gradients=1.0,
                                   learning_rate=tf.constant(3e-3, dtype=tf.float32))

    def build(self, input_shape):
        pass

    def emission(self, z_t: tf.Tensor) -> tf.Tensor:
        """
        The network x_t = ϴ(z_t)
        defines the output distribution
            p_θ(x_t | z_t)

        Args:
            z_t:

        Returns:
            # σ_θt
            μ_θt:
        """
        pass

    def transition(self, z_p: tf.Tensor, u_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The network z_t = Γ(z_p, u_t)
        defines the distribution of the latent state given the previous state and current input
            p_γ(z_t | z_p, u_t)

        Args:
            z_p:
            u_t:

        Returns:
            σ_γt
            μ_γt
        """
        pass

    def inference(self, z_p: tf.Tensor, u_t: tf.Tensor, x_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The network z_t = ϕ(z_p, u_t, x_t)
        defines the approximate variational distribution of the latent state using filtering (as opposed to smoothing)
            q_φ(z_t | z_p, u_t, x_t)

        Args:
            z_p:
            u_t:
            x_t:

        Returns:
            σ_φt:
            μ_φt:

        """
        pass

    def get_initial_state(self, x_1: tf.Tensor) -> tf.Tensor:
        """
        The network z_0 = Κ(x_1)
        defines the initial latent state using the first observed output.
            p_κ(z_0 | x_1)

        Args:

        Returns:
            σ_κ0:
            μ_κ0:
        """
        pass

    def sample_initial_state(self) -> tf.Tensor:
        """Samples the initial latent state from a multivariate gauss ball.

        Returns:
            z_0:
        """
        return tf.random.normal(shape=(1, self.latent_size), dtype=tf.float32)

    def inference_loop(self, u: tf.Tensor, x: tf.Tensor, z_0: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Starting with a given state, infers all subsequent states from u, x using the inference network.

        Args:
            u:
            x:
            z_0:

        Returns:
            z:
            σ_φ:
            μ_φ:


        """
        pass

    def transition_loop(self, n: int, z_0: tf.Tensor) -> tf.Tensor:
        """Starting with a given state, generates n subsequent states using the transition network.

        Args:
            n:
            z_0:

        Returns:
            z:
            σ_γ:
            μ_γ:

        """
        pass

    def loss(self) -> tf.Tensor:
        pass

    @tf.function
    def learn(self, xs: Dict[str, tf.Tensor]) -> None:
        """Training step for the generative model.

        Args:
            xs: Input tensor per column.

        Returns:
            Dictionary of loss tensors, and optimization operation.

        """
        self.xs = xs
        # Optimization step
        self.optimizer.optimize(
            loss=self.loss, variables=self.get_trainable_variables
        )

        return

    @staticmethod
    def diagonal_normal_kl_divergence(mu_1: tf.Tensor, stddev_1: tf.Tensor, mu_2: tf.Tensor, stddev_2: tf.Tensor):
        cov_1 = tf.square(stddev_1)
        cov_2 = tf.square(stddev_2)
        return 0.5 * tf.reduce_mean(
            tf.reduce_sum(
                tf.math.log(cov_2 / cov_1) + (tf.square(mu_1 - mu_2) + cov_1 - cov_2) / cov_2,
                axis=-1
            )
        )

    def get_trainable_variables(self):
        if self._trainable_variables is None:
            self._trainable_variables = self.trainable_variables
        return self._trainable_variables

    @property
    def regularization_losses(self):
        return [
            loss
            for module in [self.emission_network, self.emission_mean,
                      self.transition_network, self.transition_mean, self.transition_stddev,
                      self.inference_network, self.inference_mean, self.inference_stddev] + self.values
            for loss in module.regularization_losses
        ]
