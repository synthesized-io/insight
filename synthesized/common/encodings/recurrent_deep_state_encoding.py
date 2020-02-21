import tensorflow as tf

from .encoding import Encoding
from ..transformations import GaussianTransformation
from ..module import tensorflow_name_scoped


class RecurrentDSSEncoding(Encoding):
    """An encoder based of recurrent deep state space models."""
    def __init__(self, input_size: int, encoding_size: int, beta: float = 1.0, name='recurrent_dss_encoding'):
        super(RecurrentDSSEncoding, self).__init__(input_size=input_size, encoding_size=encoding_size, name=name)

        self.beta = beta
        self.capacity = input_size

        self.transition_rnn = tf.keras.layers.LSTM(
            units=self.capacity, return_sequences=True, return_state=True
        )
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

    def call(self, inputs, identifier=None, condition=(), return_encoding=False, dropout=0.):
        x = inputs  # shape: [bs, t, c]
        h_0 = self.transition_rnn.get_initial_state(x)  # shape: [bs, c]
        mask = tf.nn.dropout(tf.ones(shape=[x.shape[0], x.shape[1], 1], dtype=tf.float32), rate=dropout)

        transition_inputs = tf.concat([x[:, 0:1, :], x[:, 0:-1, :]], axis=1)  # shape: [bs, t, c]
        prior_hs, state1, state2 = self.transition_rnn(inputs=mask * transition_inputs, initial_state=h_0)
        mu_gamma, sigma_gamma = self.transition_network(prior_hs)  # shape: ([bs, t, e], [bs, t, e])

        inference_inputs = tf.concat([x, prior_hs], axis=2)   # shape: [bs, t, 2*c]
        posterior_hs = self.inference_rnn(inference_inputs)
        mu_phi, sigma_phi = self.inference_network(posterior_hs)  # shape: ([bs, t, e], [bs, t, e])

        kl_loss = self.diagonal_normal_kl_divergence(mu_1=mu_phi, stddev_1=sigma_phi,
                                                     mu_2=mu_gamma, stddev_2=sigma_gamma)
        tf.summary.scalar(name='kl_loss', data=kl_loss)
        self.add_loss(kl_loss)

        e_z = tf.random.normal(shape=sigma_phi.shape, dtype=tf.float32)
        z = mu_phi + e_z * sigma_phi  # shape: [bs, t, e]

        return z

    @tensorflow_name_scoped
    def sample(self, n, condition=(), identifier=None):
        pass

    @property
    def regularization_losses(self):
        return [loss for layer in [self.mean, self.stddev] for loss in layer.regularization_losses]