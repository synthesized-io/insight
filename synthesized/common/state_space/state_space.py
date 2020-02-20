from typing import Tuple, Dict
import tensorflow as tf
import pandas as pd

from ..optimizers import Optimizer
from ..values import ValueFactory, ValueOps


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
    def __init__(self, df, capacity: int, latent_size: int, name='state_space_model'):
        super(StateSpaceModel, self).__init__(name=name)
        self.capacity = capacity
        self.latent_size = latent_size
        self._trainable_variables = None

        self.value_factory = ValueFactory(df=df, capacity=capacity)
        self.value_ops = ValueOps(
            values=self.value_factory.get_values(), conditions=self.value_factory.get_conditions()
        )
        self.optimizer = Optimizer(name='optimizer', optimizer='adam', clip_gradients=1.0,
                                   learning_rate=tf.constant(3e-3, dtype=tf.float32))

        self.built = False

    def build(self, input_shape):
        pass

    def emission(self, z_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        The network x_t = ϴ(z_t)
        defines the output distribution
            p_θ(x_t | z_t)

        Args:
            z_t: [b, t, l]

        Returns:
            σ_θt
            μ_θt: [b, t, i]
        """
        mu, sigma = self.emission_network(z_t)

        return mu, sigma

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
            x_1: [b, 1, i]

        Returns:
            z_0: [b, 1, l]

        """
        mu, sigma = self.initial(x_1)
        e = self.sample_state(bs=x_1.shape[0])

        return mu + e*sigma

    def sample_state(self, bs: tf.Tensor = tf.constant(1, dtype=tf.int64)) -> tf.Tensor:
        """Samples the latent state from a multivariate gauss ball.

        Returns:
            e: [b, 1, l]

        """
        return tf.random.normal(shape=(bs, 1, self.latent_size), dtype=tf.float32)

    def sample_output(self, bs: tf.Tensor = tf.constant(1, dtype=tf.int64)) -> tf.Tensor:
        """Samples the latent state from a multivariate gauss ball.

        Returns:
            e: [b, 1, l]

        """
        return tf.random.normal(shape=(bs, 1, self.value_ops.output_size), dtype=tf.float32)

    # @tf.function
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

    def synthesize(self, n: int):
        z_0 = self.sample_state(bs=1)
        z, y = self.transition_loop(n=n, z_0=z_0)

        x = self.value_ops.value_outputs(y=y, conditions={})

        syn_df = pd.DataFrame(x)
        syn_df = self.value_factory.postprocess(df=syn_df)
        return syn_df

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
        self._trainable_variables = self.trainable_variables
        return self._trainable_variables

    @property
    def regularization_losses(self):
        raise NotImplementedError

    def get_all_values(self):
        return self.value_factory.get_values()

    def get_training_data(self, df: pd.DataFrame) -> Dict[str, tf.Tensor]:
        data = {
            name: tf.constant(df[name].to_numpy(), dtype=value.dtype)
            for value in self.get_all_values()
            for name in value.learned_input_columns()
        }
        return data
