# type: ignore
from typing import Dict

import pandas as pd
import tensorflow as tf

from .feed_forward_state_space import GaussianEncoder
from .state_space import StateSpaceModel
from ...metadata import DataFrameMeta


class RecurrentStateSpaceModel(StateSpaceModel):
    """"""
    def __init__(self, df_meta: DataFrameMeta, capacity: int, latent_size: int):
        super(RecurrentStateSpaceModel, self).__init__(
            df_meta=df_meta, capacity=capacity, latent_size=latent_size, name='recurrent_state_space_model'
        )

        self.emission_network = GaussianEncoder(
            input_size=latent_size, output_size=self.value_ops.output_size, capacity=capacity,
            num_layers=1, name='emission'
        )
        self.transition_rnn = tf.keras.layers.LSTM(units=capacity, return_sequences=True, return_state=True)
        self.transition_network = GaussianEncoder(
            input_size=capacity, output_size=latent_size, capacity=capacity,
            num_layers=1, name='transition'
        )

        self.inference_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=capacity // 2, return_sequences=True)
        )
        self.inference_network = GaussianEncoder(
            input_size=capacity, output_size=latent_size, capacity=capacity,
            num_layers=1, name='inference'
        )

    def build(self, input_shape):
        with tf.name_scope('emission'):
            self.emission_network.build(self.latent_size)
        with tf.name_scope('transition'):
            self.transition_network.build(self.capacity)
        with tf.name_scope('inference'):
            self.inference_network.build(self.capacity)
        self.built = True

    @property
    def regularization_losses(self):
        return None

    def initial_transition_state(self, shape):
        return (tf.zeros(shape=[shape[0], self.capacity], dtype=tf.float32),
                tf.zeros(shape=[shape[0], self.capacity], dtype=tf.float32))

    def loss(self, xs: Dict[str, tf.Tensor]) -> tf.Tensor:

        x = self.value_ops.unified_inputs(inputs=xs)  # [bs, t, i]
        trn_x = tf.concat([x[:, 0:1, :], x[:, 0:-1, :]], axis=1)
        mask = tf.nn.dropout(tf.ones(shape=[x.shape[0], x.shape[1], 1], dtype=tf.float32), rate=0.25)

        h_0 = self.transition_rnn.get_initial_state(trn_x)

        prior_hs, state1, state2 = self.transition_rnn(inputs=mask * trn_x, initial_state=h_0)

        mu_gamma, sigma_gamma = self.transition_network(prior_hs)

        inf_x = tf.concat([x, prior_hs], axis=2)
        inf_h_0 = self.inference_rnn.forward_layer.get_initial_state(inf_x) + \
            self.inference_rnn.backward_layer.get_initial_state(inf_x)
        posterior_hs = self.inference_rnn(inf_x, initial_state=inf_h_0)

        mu_phi, sigma_phi = self.inference_network(posterior_hs)

        kl_loss = self.diagonal_normal_kl_divergence(mu_1=mu_phi, stddev_1=sigma_phi,
                                                     mu_2=mu_gamma, stddev_2=sigma_gamma)

        e_z = tf.random.normal(shape=sigma_phi.shape, dtype=tf.float32)

        z = mu_phi + e_z * sigma_phi

        mu_theta, sigma_theta = self.emission_network(z)

        e_y = tf.random.normal(shape=sigma_theta.shape, dtype=tf.float32)

        y = mu_theta + e_y * sigma_theta
        x = self.value_ops.value_outputs(y=y[0, :, :], conditions={})

        syn_df = pd.DataFrame(x)
        syn_df = self.df_meta.postprocess(df=syn_df)
        fig = plt.figure(figsize=(16, 6))
        ax = fig.gca()
        sns.lineplot(data=syn_df, axes=ax, dashes=False)
        tf.summary.image("Training Data", data=plot_to_image(fig))
        plt.close(fig)

        reconstruction_loss = self.value_ops.reconstruction_loss(y=y, inputs=xs)
        loss = tf.add_n((kl_loss, reconstruction_loss), name='total_loss')

        tf.summary.scalar(name='kl_loss', data=kl_loss)
        tf.summary.scalar(name='reconstruction_loss', data=reconstruction_loss)
        tf.summary.scalar(name='total_loss', data=loss)
        return loss

    def transition_loop(self, n: int, z_0):
        z = []
        y = []

        mu_theta, sigma_theta = self.emission_network(z_0)
        e_y = tf.random.normal(shape=sigma_theta.shape, dtype=tf.float32)
        y_t = mu_theta + e_y * sigma_theta

        h_t, state1, state2 = self.transition_rnn(
            y_t,
            initial_state=(tf.random.normal(shape=(1, self.capacity), dtype=tf.float32),
                           tf.zeros(shape=(1, self.capacity), dtype=tf.float32))
        )

        for _ in range(n):
            mu_gamma, sigma_gamma = self.transition_network(h_t)
            e_z = tf.random.normal(shape=sigma_gamma.shape, dtype=tf.float32)
            z_t = mu_gamma + e_z * sigma_gamma
            mu_theta, sigma_theta = self.emission_network(z_t)
            e_y = tf.random.normal(shape=sigma_theta.shape, dtype=tf.float32)
            y_t = mu_theta + e_y * sigma_theta
            h_t, state1, state2 = self.transition_rnn(y_t, initial_state=(state1, state2))

            z.append(z_t)
            y.append(y_t)

        return tf.concat(z, axis=1), tf.concat(y, axis=1)

    def synthesize(self, n: int):
        z_0 = self.sample_state(bs=1)
        z, y = self.transition_loop(n=n, z_0=z_0)

        x = self.value_ops.value_outputs(y=y, conditions={})
        syn_df = pd.DataFrame(x)
        syn_df = self.df_meta.postprocess(df=syn_df)
        return syn_df


if __name__ == '__main__':
    """Testing the RSSM on simple sinusoidal data."""
    import io
    import warnings
    from datetime import datetime

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from synthesized.common.util import record_summaries_every_n_global_steps

    from ...metadata import MetaExtractor
    warnings.filterwarnings('ignore', module='pandas|sklearn')

    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    value = np.sin(np.linspace(0, 20, 200))
    df = pd.DataFrame(dict(
        a=value,
        b=-value
    ))

    df_meta = MetaExtractor.extract(df)
    rssm = RecurrentStateSpaceModel(df_meta=df_meta, capacity=8, latent_size=4)

    df_train = rssm.df_meta.preprocess(df)
    data = rssm.get_training_data(df_train)

    global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    tf.summary.experimental.set_step(global_step)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = f"logs/rdssm/{stamp}"

    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()
    rssm.build(None)
    with record_summaries_every_n_global_steps(5, global_step=global_step):
        for j in range(50):
            for i in range(10):
                if j == i == 0:
                    tf.summary.trace_on(graph=True, profiler=False)
                indices = [np.random.randint(0, 8) for _ in range(8)]
                data2 = {
                    k: tf.constant(np.stack([v[:200] for idx in indices]), dtype=tf.float32)
                    for k, v in data.items()
                }
                rssm.learn(xs=data2)
                global_step.assign_add(1)
                writer.flush()
                if j == i == 0:
                    tf.summary.trace_export(name="Learn", step=global_step)
                    tf.summary.trace_off()

            syn = rssm.synthesize(200)
            fig = plt.figure(figsize=(16, 6))
            ax = fig.gca()
            sns.lineplot(data=syn, axes=ax, dashes=False)
            tf.summary.image("Synthesized Data", data=plot_to_image(fig))
            plt.close(fig)
