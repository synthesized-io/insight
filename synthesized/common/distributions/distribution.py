import tensorflow as tf
from tensorflow_probability import distributions as tfd

from ..transformations.linear import LinearTransformation

# TensorFlow distribution implementations
tf_distributions = dict(
    deterministic=tfd.Deterministic,
    normal=tfd.Normal
)


class Distribution(tf.keras.layers.Layer):
    """Parametrized distribution, either directly or by a neural network."""

    def __init__(
        self, name: str,
        # Input and output size
        input_size: int, output_size: int,
        # Distribution: "deterministic", "normal"
        distribution: str, beta: float = None, encode: bool = False
    ):
        super().__init__(name=name)
        self.beta = beta
        self.encode = encode
        # Output size
        self.output_size = output_size

        # Distribution
        if distribution not in tf_distributions:
            raise NotImplementedError
        self.distribution = distribution

        # Distribution-specific parameters
        if self.distribution == 'deterministic':
            # Deterministic Dirac distribution: value
            self.mean = LinearTransformation(
                name='mean', input_size=input_size, output_size=output_size
            )
            self.distr_trafos = [self.mean]
        elif self.distribution == 'normal':
            # Normal distribution: mean and variance
            self.mean = LinearTransformation(
                name='mean', input_size=input_size, output_size=output_size
            )
            self.stddev = LinearTransformation(
                name='stddev', input_size=input_size, output_size=output_size
            )
            self.distr_trafos = [self.mean, self.stddev]
        else:
            raise NotImplementedError

    def specification(self):
        spec = super().specification()
        spec.update(
            output_size=self.output_size, distribution=self.distribution,
            distr_trafos=[trafo.specification() for trafo in self.distr_trafos]
        )
        return spec

    def size(self):
        return self.output_size

    def build(self, input_shape):
        if self.distribution == 'deterministic':
            self.mean.build(input_shape=input_shape)
        elif self.distribution == 'normal':
            self.mean.build(input_shape=input_shape)
            self.stddev.build(input_shape=input_shape)
        else:
            raise NotImplementedError

        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tfd.Distribution:
        # Distribution arguments
        if self.distribution == 'deterministic':
            # Deterministic distribution
            loc = self.mean(inputs)
            kwargs.update(loc=loc)
        elif self.distribution == 'normal':
            # Normal distribution
            loc = self.mean(inputs)
            scale = tf.exp(x=self.stddev(inputs))
            kwargs.update(dict(loc=loc, scale=scale))
        else:
            raise NotImplementedError

        # TensorFlow distribution
        p = tf_distributions[self.distribution](validate_args=True, allow_nan_stats=False, **kwargs)

        if self.encode:
            if p.reparameterization_type is not tfd.FULLY_REPARAMETERIZED:
                raise NotImplementedError

            # KL-divergence loss
            kldiv = tfd.kl_divergence(distribution_a=p, distribution_b=self.prior(), allow_nan_stats=False)
            kldiv = tf.reduce_sum(input_tensor=kldiv, axis=1)
            kldiv = tf.reduce_mean(input_tensor=kldiv, axis=0)
            kl_loss = tf.multiply(self.beta, kldiv, name='kl_loss')
            tf.summary.scalar(name='kl-loss', data=kl_loss)
            self.add_loss(kl_loss)

        return p

    def prior(self) -> tfd.Distribution:
        return Distribution.get_prior(distribution=self.distribution, size=self.output_size)

    @staticmethod
    def get_prior(distribution, size) -> tfd.Distribution:
        if distribution not in tf_distributions:
            raise NotImplementedError

        # Distribution arguments
        if distribution == 'deterministic':
            # Deterministic distribution
            loc = tf.zeros(shape=(size,))
            kwargs = dict(loc=loc)
        elif distribution == 'normal':
            # Normal distribution
            loc = tf.zeros(shape=(size,))
            scale = tf.ones(shape=(size,))
            kwargs = dict(loc=loc, scale=scale)
        else:
            raise NotImplementedError

        # TensorFlow distribution
        p = tf_distributions[distribution](**kwargs)

        return p

    @property
    def regularization_losses(self):
        return [loss for module in self.distr_trafos for loss in module.regularization_losses]
