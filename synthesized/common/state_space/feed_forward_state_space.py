from typing import Tuple

import pandas as pd
import tensorflow as tf

from .state_space import StateSpaceModel
from ..transformations import Transformation, MlpTransformation, DenseTransformation


class FeedForwardStateSpaceModel(StateSpaceModel):
    """A Deep State Space model using only feed forward networks"""
    def __init__(self, df: pd.DataFrame, capacity: int, latent_size: int, name: str = 'ff_state_space_model'):
        super(FeedForwardStateSpaceModel, self).__init__(df=df, capacity=capacity, latent_size=latent_size,
                                                         name=name)

        self.emission_network = GaussianEncoder(
            input_size=latent_size, output_size=self.value_ops.output_size, capacity=capacity,
            num_layers=1, name='emission'
        )
        self.transition_network = GaussianEncoder(
            input_size=latent_size+self.value_ops.output_size, output_size=latent_size, capacity=capacity,
            num_layers=3, name='transition'
        )
        self.inference_network = GaussianEncoder(
            input_size=latent_size + 2*self.value_ops.output_size, output_size=latent_size, capacity=capacity,
            num_layers=3, name='inference'
        )
        self.initial_network = GaussianEncoder(
            input_size=self.value_ops.input_size, output_size=latent_size, capacity=capacity,
            num_layers=1, name='initial'
        )

    def build(self, input_shape):
        with tf.name_scope('emission'):
            self.emission_network.build(self.latent_size)
        with tf.name_scope('transition'):
            self.transition_network.build(self.latent_size + self.value_ops.output_size)
        with tf.name_scope('inference'):
            self.inference_network.build(self.latent_size + 2*self.value_ops.output_size)
        with tf.name_scope('initial'):
            self.initial_network.build(self.value_ops.input_size)
        self.built = True

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
        z = [z_0, ]
        mu, sigma = [], []

        for i in range(u.shape[1]):

            mu_t, sigma_t = self.inference(z_p=z[i], u_t=u[:, i:i+1, :], x_t=x[:, i:i+1, :])
            e = self.sample_state(bs=u.shape[0])
            z.append(mu_t + sigma_t*e)
            mu.append(mu_t)
            sigma.append(sigma_t)

        z_f = tf.concat(values=z[1:], axis=1, name='z_phi')
        mu_f = tf.concat(values=mu, axis=1, name='mu_phi')
        sigma_f = tf.concat(values=sigma, axis=1, name='sigma_phi')

        return z_f, mu_f, sigma_f

    def transition_loop(self, n: int, z_0: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Starting with a given state, generates n subsequent states using the transition network.

                Args:
                    n: []
                    z_0: [b, 1, l]

                Returns:
                    z: [b, t, l]
                    x: [b, t, i]

        """
        mu_theta_1, sigma_theta_1 = self.emission(z_0)
        x_0 = mu_theta_1 + sigma_theta_1 * self.sample_output(bs=z_0.shape[0])

        z, x = [z_0], [x_0]

        for i in range(n):
            mu_t, sigma_t = self.transition(z_p=z[i], u_t=x[i])
            z_t = mu_t + sigma_t*self.sample_state(bs=z_0.shape[0])

            mu_theta_t, sigma_theta_t = self.emission(z_t)
            x_t = mu_theta_t + sigma_theta_t*self.sample_output(bs=z_0.shape[0])

            z.append(z_t)
            x.append(x_t)

        z_f = tf.concat(values=z[1:], axis=1, name='z_gamma')
        x_f = tf.concat(values=x[1:], axis=1, name='x_gamma')

        return z_f, x_f

    def loss(self) -> tf.Tensor:
        x = self.value_ops.unified_inputs(inputs=self.xs)

        z_0 = self.get_initial_state(x_1=x[:, 0:1, :])
        mu_theta_0, sigma_theta_0 = self.emission(z_t=z_0)
        u_1 = mu_theta_0 + sigma_theta_0 * self.sample_output(bs=z_0.shape[0])

        u = tf.concat((u_1, x[:, :-1, :]), axis=1, name='u')

        z, mu_phi, sigma_phi = self.inference_loop(u=u, x=x, z_0=z_0)
        z_p = tf.concat((z_0, z[:, :-1, :]), axis=1, name='z_p')

        mu_gamma, sigma_gamma = self.transition(z_p=z_p, u_t=u)
        mu_theta, sigma_theta = self.emission(z_t=z)

        y = mu_theta + sigma_theta * tf.random.normal(shape=mu_theta.shape)

        # Losses: kl, reconstruction and total
        kl_loss = self.diagonal_normal_kl_divergence(mu_1=mu_phi, stddev_1=sigma_phi,
                                                     mu_2=mu_gamma, stddev_2=sigma_gamma)
        normal_kl_loss = self.diagonal_normal_kl_divergence(
            mu_1=mu_phi, stddev_1=sigma_phi, mu_2=tf.zeros(shape=mu_phi.shape, dtype=tf.float32),
            stddev_2=tf.ones(shape=sigma_phi.shape, dtype=tf.float32)
        )
        init_kl_loss = self.diagonal_normal_kl_divergence(
            mu_1=mu_theta_0, stddev_1=sigma_theta_0, mu_2=tf.zeros(shape=mu_theta_0.shape, dtype=tf.float32),
            stddev_2=tf.ones(shape=sigma_theta_0.shape, dtype=tf.float32)
        )
        reconstruction_loss = self.value_ops.reconstruction_loss(y=y, inputs=self.xs)
        loss = tf.add_n((kl_loss, init_kl_loss, reconstruction_loss, normal_kl_loss), name='total_loss')

        tf.summary.scalar(name='kl_loss', data=kl_loss)
        tf.summary.scalar(name='normal_kl_loss', data=normal_kl_loss)
        tf.summary.scalar(name='init_kl_loss', data=init_kl_loss)
        tf.summary.scalar(name='reconstruction_loss', data=reconstruction_loss)
        tf.summary.scalar(name='total_loss', data=loss)
        return loss

    def regularization_losses(self):
        pass


class GaussianEncoder(Transformation):
    def __init__(
            self, input_size: int, output_size: int, capacity: int, num_layers: int, name: str = 'gaussian-encoder'
    ):
        super(GaussianEncoder, self).__init__(name=name, input_size=input_size, output_size=output_size)

        self.network = MlpTransformation(
            name='mlp-encoder', input_size=input_size, layer_sizes=[capacity for _ in range(num_layers)],
            batchnorm=False
        )
        self.mean = DenseTransformation(
            name='mean', input_size=capacity,
            output_size=output_size, batchnorm=False, activation='none'
        )
        self.stddev = DenseTransformation(
            name='stddev', input_size=capacity,
            output_size=output_size, batchnorm=False, activation='softplus'
        )

    def build(self, input_shape):
        self.network.build(input_shape)
        self.mean.build(self.network.output_size)
        self.stddev.build(self.network.output_size)
        self.built = True

    def call(self, inputs, **kwargs):
        z = self.network(inputs, **kwargs)
        mu, sigma = self.mean(z, **kwargs), self.stddev(z, **kwargs)
        tf.summary.histogram(name='mean', data=mu)
        tf.summary.histogram(name='stddev', data=sigma)
        return mu, sigma


if __name__ == '__main__':
    """Testing the FFSSM on simple sinusoidal data."""
    import warnings
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    from synthesized.common.util import record_summaries_every_n_global_steps
    warnings.filterwarnings('ignore', module='pandas|sklearn')

    df = pd.DataFrame(dict(
        a=np.sin(np.linspace(0, 200, 3000)),
        b=-np.sin(np.linspace(0, 200, 3000))
    ))

    ffssm = FeedForwardStateSpaceModel(df=df, capacity=8, latent_size=4)

    fig = plt.figure(figsize=(16, 6))
    ax = fig.gca()
    sns.lineplot(data=df.loc[:200, :], axes=ax, dashes=False)
    plt.savefig('logs/ffdssm/original.png', dpi=100)
    plt.close(fig)

    df_train = ffssm.value_factory.preprocess(df)
    data = ffssm.get_training_data(df_train)

    global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    tf.summary.experimental.set_step(global_step)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = f"logs/ffdssm/{stamp}"

    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()
    ffssm.build(None)
    with record_summaries_every_n_global_steps(5, global_step=global_step):
        for i in range(50):
            for i in range(10):
                indices = [np.random.randint(0, len(df) - 200) for _ in range(64)]
                data2 = {k: np.array([v[:, idx:idx + 200] for idx in indices]) for k, v in data.items()}
                ffssm.learn(xs=data2)
                global_step.assign_add(1)
                writer.flush()

            syn = ffssm.synthesize(200)
            fig = plt.figure(figsize=(16, 6))
            ax = fig.gca()
            sns.lineplot(data=syn, axes=ax, dashes=False)
            plt.savefig('logs/ffdssm/synthesized.png', dpi=100)
            plt.close(fig)
