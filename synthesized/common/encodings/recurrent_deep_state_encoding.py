from typing import Callable

import tensorflow as tf

from .encoding import Encoding
from ..transformations import GaussianTransformation
from ..module import tensorflow_name_scoped


class RecurrentDSSEncoding(Encoding):
    """An encoder based on recurrent deep state space models."""
    def __init__(self, input_size: int, encoding_size: int, emission_function: Callable[[tf.Tensor], tf.Tensor],
                 beta: float = 1.0, num_transition_layers: int = 2, name='recurrent_dss_encoding'):
        super(RecurrentDSSEncoding, self).__init__(input_size=input_size, encoding_size=encoding_size, name=name)

        self.beta = beta
        self.capacity = input_size
        self.emission = emission_function
        self.num_transition_layers = num_transition_layers

        cells = [
            tf.keras.layers.LSTMCell(units=self.capacity) for _ in range(self.num_transition_layers)
        ]

        self.transition_rnn = tf.keras.layers.RNN(cell=cells, return_sequences=True, return_state=True)

        self.transition_network = GaussianTransformation(
            input_size=self.capacity, output_size=self.encoding_size, name='transition_distribution'
        )
        self.inference_rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=self.capacity//2, return_sequences=True)
        )
        self.inference_network = GaussianTransformation(
            input_size=self.capacity, output_size=self.encoding_size, name='inference_distribution'
        )

    def build(self, input_shape):
        self.transition_network.build(input_shape=(None, None, self.capacity))
        self.transition_rnn.build(input_shape=(None, None, self.input_size))

        self.inference_network.build(input_shape=(None, None, self.capacity))
        self.inference_rnn.build(input_shape=(None, None, self.input_size+self.capacity))

        self.built = True

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    def call(self, inputs, identifier=None, condition=(), return_encoding=False, series_dropout=0.,
             n_forcast=0):
        x = inputs  # shape: [bs, t, c]
        h_0 = self.transition_rnn.get_initial_state(x)  # shape: ( ([bs, c], [bs, c]), ([bs, c], [bs, c]]) )

        mask = tf.nn.dropout(tf.ones(shape=[x.shape[0], x.shape[1], 1], dtype=tf.float32), rate=series_dropout)

        transition_inputs = tf.concat([x[:, 0:1, :], x[:, 0:-1, :]], axis=1)  # shape: [bs, t, c]
        outputs = self.transition_rnn(inputs=mask * transition_inputs, initial_state=h_0)
        prior_hs = outputs[0]
        state = outputs[1:]

        mu_gamma, sigma_gamma = self.transition_network(prior_hs)  # shape: ([bs, t, e], [bs, t, e])

        inference_inputs = tf.concat([x, prior_hs], axis=2)   # shape: [bs, t, 2*c]
        posterior_hs = self.inference_rnn(inference_inputs)
        mu_phi, sigma_phi = self.inference_network(posterior_hs)  # shape: ([bs, t, e], [bs, t, e])

        kl_loss = self.diagonal_normal_kl_divergence(mu_1=mu_phi, stddev_1=sigma_phi,
                                                     mu_2=mu_gamma, stddev_2=sigma_gamma)
        transition_regularization = self.diagonal_normal_kl_divergence(mu_1=mu_phi, stddev_1=sigma_phi)

        tf.summary.scalar(name='kl_loss', data=kl_loss)
        tf.summary.scalar(name='transition_regularization', data=transition_regularization)

        kl_loss = kl_loss+0.005*transition_regularization

        self.add_loss(kl_loss)

        e_z = tf.random.normal(shape=sigma_phi.shape, dtype=tf.float32)
        z = mu_phi + e_z * sigma_phi  # shape: [bs, t, e]

        if n_forcast > 0:
            z_fc, y_fc = self.transition_loop(n=n_forcast, z_0=z[:, -1:, :], initial_state=state)
            z = tf.concat((z, z_fc), axis=1)

        return z

    def sample_state(self, bs: int = 1) -> tf.Tensor:
        """Samples the latent state from a multivariate gauss ball.

        Returns:
            e: [b, 1, l]

        """
        return tf.zeros(shape=(bs, 1, self.encoding_size), dtype=tf.float32)

    def transition_loop(self, n: int, z_0, initial_state=None):
        z = []
        y = []
        if initial_state is None:
            initial_state = [(
                tf.random.normal(shape=(1, self.capacity), dtype=tf.float32),
                tf.zeros(shape=(1, self.capacity), dtype=tf.float32)
            ) for _ in range(self.num_transition_layers)]

        y_t = self.emission(z_0)
        outputs = self.transition_rnn(y_t, initial_state=initial_state)

        h_t = outputs[0]
        state = outputs[1:]

        for _ in range(n):
            mu_gamma, sigma_gamma = self.transition_network(h_t)
            e_z = tf.random.normal(shape=sigma_gamma.shape, dtype=tf.float32)
            z_t = mu_gamma + e_z * sigma_gamma
            y_t = self.emission(z_t)
            outputs = self.transition_rnn(y_t, initial_state=state)
            h_t = outputs[0]
            state = outputs[1:]

            z.append(z_t)
            y.append(y_t)

        return tf.concat(z, axis=1), tf.concat(y, axis=1)

    @tensorflow_name_scoped
    def sample(self, n, condition=(), identifier=None):
        if identifier is not None:
            z_0 = tf.broadcast_to(identifier, shape=(1, 1, self.capacity))
        else:
            z_0 = self.sample_state(bs=1)
        z, y = self.transition_loop(n=n, z_0=z_0)

        return z

    @property
    def regularization_losses(self):
        return [
            loss for layer in [self.transition_network, self.inference_network] for loss in layer.regularization_losses
        ]
