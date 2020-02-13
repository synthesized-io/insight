from typing import Tuple, Dict, Optional

import tensorflow as tf

from ..optimizers import Optimizer
from ..values import ValueFactory
from ..transformations import Transformation


class StateSpaceModel(tf.Module):
    """The base class for state space models

    Networks:
        ϴ: emission
        Γ: transmission
        ϕ: inference
        Κ: initialize


    Dimensions:
        b: batch size
        i: input size
        l: latent size
        c: capacity
        t: time length

    """
    def __init__(self, df, capacity, latent_size, name='state_space_model'):
        super(StateSpaceModel, self).__init__(name=name)
        self.capacity = capacity
        self.latent_size = latent_size
        self._trainable_variables = None

        self.value_factory = ValueFactory(df, capacity=capacity)
        self.optimizer = Optimizer(name='optimizer', optimizer='adam', clip_gradients=1.0,
                                   learning_rate=tf.constant(3e-3, dtype=tf.float32))

        self.emission_network: Optional[Transformation] = None
        self.transition_network: Optional[Transformation] = None
        self.inference_network: Optional[Transformation] = None
        self.initial_network: Optional[Transformation] = None
        self.built = False

    def build(self, input_shape):
        pass

    def emission(self, z_t: tf.Tensor) -> tf.Tensor:
        """
        The network x_t = ϴ(z_t)
        defines the output distribution
            p_θ(x_t | z_t)

        Args:
            z_t: [b, t, l]

        Returns:
            # σ_θt
            μ_θt: [b, t, i]
        """
        mu = self.emission_network(z_t)

        return mu

    def transition(self, z_p: tf.Tensor, u_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The network z_t = Γ(z_p, u_t)
        defines the distribution of the latent state given the previous state and current input
            p_γ(z_t | z_p, u_t)

        Args:
            z_p: [b, t, l]
            u_t: [b, t, i]

        Returns:
            σ_γt: [b, t, l]
            μ_γt: [b, t, l]
        """
        inputs = tf.concat([z_p, u_t], axis=-1)
        mu, sigma = self.transition_network(inputs)

        return mu, sigma

    def inference(self, z_p: tf.Tensor, u_t: tf.Tensor, x_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The network z_t = ϕ(z_p, u_t, x_t)
        defines the approximate variational distribution of the latent state using filtering (as opposed to smoothing)
            q_φ(z_t | z_p, u_t, x_t)

        Args:
            z_p: [b, t, l]
            u_t: [b, t, i]
            x_t: [b, t, i]

        Returns:
            σ_φt: [b, t, l]
            μ_φt: [b, t, l]

        """
        inputs = tf.concat([z_p, u_t, x_t], axis=-1)
        mu, sigma = self.inference_network(inputs)

        return mu, sigma

    def initial(self, x_1: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The network z_0 = Κ(x_1)
        defines the initial latent state using the first observed output.
            p_κ(z_0 | x_1)

        Args:
            x_1: [b, 1, i]

        Returns:
            σ_κ0: [b, 1, l]
            μ_κ0: [b, 1, l]
        """
        mu, sigma = self.initial_network(x_1)

        return mu, sigma

    def get_initial_state(self, x_1: tf.Tensor) -> tf.Tensor:
        """

        Args:
            x_1:

        Returns:
            z_0: [b, 1, l]

        """
        pass

    def sample_initial_state(self) -> tf.Tensor:
        """Samples the initial latent state from a multivariate gauss ball.

        Returns:
            z_0: [b, 1, l]

        """
        return tf.random.normal(shape=(1, self.latent_size), dtype=tf.float32)

    def inference_loop(self, u: tf.Tensor, x: tf.Tensor, z_0: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Starting with a given state, infers all subsequent states from u, x using the inference network.

        Args:
            u: [b, t, i]
            x: [b, t, i]
            z_0: [b, 1, l]

        Returns:
            z: [b, t, l]
            σ_φ: [b, t, l]
            μ_φ: [b, t, l]

        """
        pass

    def transition_loop(self, n: int, z_0: tf.Tensor) -> tf.Tensor:
        """Starting with a given state, generates n subsequent states using the transition network.

        Args:
            n: []
            z_0: [b, 1, l]

        Returns:
            z: [b, t, l]
            σ_γ: [b, t, l]
            μ_γ: [b, t, l]

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
        raise NotImplementedError
