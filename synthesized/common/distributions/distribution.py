import tensorflow as tf
from tensorflow_probability import distributions as tfd

from ..module import tensorflow_name_scoped
from ..module import Module


# TensorFlow distribution implementations
tf_distributions = dict(
    deterministic=tfd.Deterministic,
    normal=tfd.Normal
)


class Distribution(Module):
    """Parametrized distribution, either directly or by a neural network."""

    def __init__(
        self, name: str,
        # Input and output size
        input_size: int, output_size: int,
        # Distribution: "deterministic", "normal"
        distribution: str,
        # Parametrization specification excluding name and input_size
        parametrization: dict = None
    ):
        super().__init__(name=name)

        # Output size
        self.output_size = output_size

        # Distribution
        if distribution not in tf_distributions:
            raise NotImplementedError
        self.distribution = distribution

        # Parametrization
        if parametrization is None:
            # Direct parametrization
            self.parametrization = None
        else:
            # Neural network parametrization
            self.parametrization = self.add_module(
                name='parametrization', input_size=input_size, **parametrization
            )
            input_size = self.parametrization.size()

        # Distribution-specific parameters
        if self.distribution == 'deterministic':
            # Deterministic Dirac distribution: value
            self.loc = self.add_module(
                module='linear', name='loc', input_size=input_size, output_size=output_size
            )
            self.distr_trafos = [self.loc]
        elif self.distribution == 'normal':
            # Normal distribution: mean and variance
            self.loc = self.add_module(
                module='linear', name='loc', input_size=input_size, output_size=output_size
            )
            self.scale = self.add_module(
                module='linear', name='scale', input_size=input_size, output_size=output_size
            )
            self.distr_trafos = [self.loc, self.scale]
        else:
            raise NotImplementedError

    def specification(self):
        spec = super().specification()
        spec.update(
            output_size=self.output_size, distribution=self.distribution,
            distr_trafos=[trafo.specification() for trafo in self.distr_trafos],
            parametrization=self.parametrization.specification()
        )
        return spec

    def size(self):
        return self.output_size

    @tensorflow_name_scoped
    def parametrize(self, x: tf.Tensor) -> tfd.Distribution:
        if self.parametrization is not None:
            # Neural network parametrization
            x = self.parametrization.transform(x=x)

        # Distribution arguments
        if self.distribution == 'deterministic':
            # Deterministic distribution
            loc = self.loc.transform(x=x)
            kwargs = dict(loc=loc)
        elif self.distribution == 'normal':
            # Normal distribution
            loc = self.loc.transform(x=x)
            scale = tf.exp(x=self.scale.transform(x=x))
            kwargs = dict(loc=loc, scale=scale)
        else:
            raise NotImplementedError

        # TensorFlow distribution
        p = tf_distributions[self.distribution](validate_args=True, allow_nan_stats=False, **kwargs)

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
