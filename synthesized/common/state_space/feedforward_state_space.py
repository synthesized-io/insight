import pandas as pd
import tensorflow as tf

from .state_space import StateSpaceModel
from ..values import ValueFactory, ValueOps
from ..transformations import Transformation, MlpTransformation, DenseTransformation


class FeedForwardStateSpaceModel(StateSpaceModel):
    """A Deep State Space model using only feed forward networks"""
    def __init__(self, df: pd.DataFrame, capacity: int, latent_size: int, name: str = 'ff_state_space_model'):
        super(FeedForwardStateSpaceModel, self).__init__(df=df, capacity=capacity, latent_size=latent_size,
                                                         name=name)

        self.emission_network = GaussianEncoder(
            input_size=latent_size, output_size=self.value_ops.output_size, capacity=capacity,
            num_layers=1
        )
        self.transition_network = GaussianEncoder(
            input_size=latent_size+self.value_ops.output_size, output_size=latent_size, capacity=capacity,
            num_layers=4
        )
        self.inference_network = GaussianEncoder(
            input_size=latent_size + 2*self.value_ops.output_size, output_size=latent_size, capacity=capacity,
            num_layers=4
        )
        self.initial_network = GaussianEncoder(
            input_size=self.value_ops.input_size, output_size=latent_size, capacity=capacity,
            num_layers=1
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

    def loss(self) -> tf.Tensor:
        x = self.value_ops.unified_inputs(inputs=self.xs)

        z_0 = self.get_initial_state(x_1=x[:, 0:1, :])

        mu_theta_0, sigma_theta_0 = self.emission(z_t=z_0)
        u_1 = mu_theta_0 + sigma_theta_0 * self.sample_output(bs=z_0.shape[0])

        u = tf.concat((u_1, x[:,:-1,:]), axis=1, name='u')

        z, mu_phi, sigma_phi = self.inference_loop(u=u, x=x, z_0=z_0)

        z_p = tf.concat((z_0, z[:, :-1, :]), axis=1, name='z_p')

        mu_gamma, sigma_gamma = self.transition(z_p=z_p, u_t=u)

        kl_loss = self.diagonal_normal_kl_divergence(mu_1=mu_phi, stddev_1=sigma_phi,
                                                     mu_2=mu_gamma, stddev_2=sigma_gamma)
        tf.summary.scalar(name='kl_loss', data=kl_loss)

        init_kl_loss = self.diagonal_normal_kl_divergence(
            mu_1=mu_theta_0, stddev_1=sigma_theta_0, mu_2=tf.zeros(shape=mu_theta_0.shape, dtype=tf.float32),
            stddev_2=tf.ones(shape=sigma_theta_0.shape, dtype=tf.float32)
        )
        tf.summary.scalar(name='init_kl_loss', data=init_kl_loss)

        mu_theta, sigma_theta = self.emission(z_t=z)
        y = mu_theta + sigma_theta * tf.random.normal(shape=mu_theta.shape)

        reconstruction_loss = self.value_ops.reconstruction_loss(y=y, inputs=self.xs)
        tf.summary.scalar(name='reconstruction_loss', data=reconstruction_loss)
        loss = tf.add_n((kl_loss, init_kl_loss, reconstruction_loss), name='total_loss')
        tf.summary.scalar(name='total_loss', data=loss)
        return loss

    def get_all_values(self):
        return self.value_factory.get_values()

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
        z = self.network(inputs)
        mu, sigma = self.mean(z), self.stddev(z)

        return mu, sigma


if __name__ == '__main__':
    import warnings
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    warnings.filterwarnings('ignore', module='pandas|sklearn')

    df = pd.read_csv(open('data/time-series/air-quality.csv', 'r')).dropna()
    del(df['Date'])
    del(df['Time'])

    ffssm = FeedForwardStateSpaceModel(df=df, capacity=8, latent_size=8)

    fig = plt.figure(figsize=(16,6))
    ax = fig.gca()
    sns.lineplot(data=df.loc[:200, :], axes=ax, dashes=False)
    plt.savefig('logs/ffdssm/original.png', dpi=100)
    plt.close(fig)

    df_train = ffssm.value_factory.preprocess(df)

    data = ffssm.get_training_data(df_train)
    data = {k: v[:200] for k, v in data.items()}

    global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    tf.summary.experimental.set_step(global_step)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = f"logs/ffdssm/{stamp}"

    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()
    ffssm.build(None)
    for i in range(20):
        for i in range(20):
            ffssm.learn(xs=data)
            global_step.assign_add(1)
            writer.flush()

            syn = ffssm.synthesize(200)

            fig = plt.figure(figsize=(16, 6))
            ax = fig.gca()
            sns.lineplot(data=syn, axes=ax, dashes=False)
            plt.savefig('logs/ffdssm/synthesized.png', dpi=100)
            plt.close(fig)
