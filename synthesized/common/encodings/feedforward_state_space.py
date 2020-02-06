import tensorflow as tf
from tensorflow import keras

from .encoding import Encoding
from ..module import tensorflow_name_scoped


class FeedForwardDSSMEncoding(Encoding):
    def __init__(self, name, input_size, encoding_size, lstm_size, beta=None):
        super(FeedForwardDSSMEncoding, self).__init__(name=name, input_size=input_size, encoding_size=encoding_size)
        # -- KL loss weight
        self.beta = beta

        # -- Transition model
        # --- Initial input vector
        self.init_input = tf.Variable(tf.random.normal(shape=[self.input_size, ]), name='init_input', trainable=False)

        # --- Transition distribution
        self.transition_mean = keras.layers.Dense(self.encoding_size)
        self.transition_stddev = keras.Sequential([keras.layers.Dense(self.encoding_size),
                                                   keras.layers.Activation("softplus")])

        # -- Variational Posterior
        # --- BiLSTM encoder
        self.lstm_size = lstm_size
        self.posterior_encoder = keras.layers.Bidirectional(keras.layers.LSTM(units=self.lstm_size,
                                                                              return_sequences=True),
                                                            input_shape=())

        # --- Variational distribution
        self.posterior_mean = keras.layers.Dense(self.encoding_size)
        self.posterior_stddev = keras.Sequential([keras.layers.Dense(self.encoding_size),
                                                  keras.layers.Activation("softplus")])

    def init_h(self):
        return tf.expand_dims(self.init_input, 0)

    @tensorflow_name_scoped
    def build(self, input_shape):
        self.posterior_encoder.build(input_shape=(None, None, self.input_size))
        self.posterior_mean.build(input_shape=(None, 2*self.lstm_size))
        self.posterior_stddev.build(input_shape=(None, 2*self.lstm_size))

        self.transition_mean.build(input_shape=(None, self.encoding_size+self.input_size))
        self.transition_stddev.build(input_shape=(None, self.encoding_size+self.input_size))
        self.built = True

    def specification(self):
        spec = super().specification()
        spec.update(beta=self.beta)
        return spec

    def size(self):
        return self.encoding_size

    @staticmethod
    def diagonal_normal_kl_divergence(mu_1, stddev_1, mu_2, stddev_2):
        cov_1 = tf.square(stddev_1)
        cov_2 = tf.square(stddev_2)
        return 0.5*tf.reduce_mean(tf.reduce_sum(tf.math.log(cov_2/cov_1) + (tf.square(mu_1-mu_2) + cov_1 - cov_2)/cov_2,
                                                axis=1), axis=0)

    def call(self, x, condition=(),):
        inputs = x
        expanded_inputs = tf.expand_dims(inputs, 0)

        posterior_hs = tf.squeeze(self.posterior_encoder(inputs=tf.nn.dropout(expanded_inputs, rate=0.4)), axis=0)

        posterior_means = self.posterior_mean(posterior_hs)
        posterior_stddevs = self.posterior_stddev(posterior_hs)
        posterior_eps = tf.random.normal(
            shape=tf.shape(input=posterior_means), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        zs = posterior_means + posterior_stddevs*posterior_eps
        prior_contexts = tf.concat([zs, inputs], axis=1)
        prior_means = self.transition_mean(prior_contexts)
        prior_stddevs = self.transition_stddev(prior_contexts)
        kl_loss = self.diagonal_normal_kl_divergence(mu_1=posterior_means, stddev_1=posterior_stddevs,
                                                     mu_2=prior_means, stddev_2=prior_stddevs)
        self.add_loss(kl_loss)
        tf.summary.scalar(name='kl-loss', data=kl_loss)

        return zs

    @tensorflow_name_scoped
    def sample(self, inputs, state, condition=()):
        prior_contexts = tf.concat([state, inputs], axis=1)
        prior_means = self.transition_mean(prior_contexts)
        prior_stddevs = self.transition_stddev(prior_contexts)
        prior_eps = tf.random.normal(
            shape=tf.shape(input=prior_means), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None
        )
        return prior_means + prior_stddevs*prior_eps

