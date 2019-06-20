import tensorflow as tf
from tensorflow_probability import distributions as tfd

from ..module import tensorflow_name_scoped
from ..transformations import Transformation


tf_distributions = dict(
    deterministic=tfd.Deterministic,
    normal=tfd.Normal
)


class Distribution(Transformation):

    def __init__(
        self, name: str, input_size: int, output_size: int, distribution: str,
        parametrization: dict = None
    ):
        super().__init__(name=name, input_size=input_size, output_size=output_size)

        assert distribution in tf_distributions
        self.distribution = distribution
        self.distribution_cls = tf_distributions[self.distribution]

        if parametrization is None:
            self.parametrization = None
        else:
            self.parametrization = self.add_module(
                name='parametrization', input_size=input_size, **parametrization
            )
            input_size = self.parametrization.size()

        if self.distribution == 'deterministic':
            self.loc = self.add_module(
                module='linear', name='loc', input_size=input_size, output_size=output_size
            )
        elif self.distribution == 'normal':
            self.loc = self.add_module(
                module='linear', name='loc', input_size=input_size, output_size=output_size
            )
            self.scale = self.add_module(
                module='linear', name='scale', input_size=input_size, output_size=output_size
            )
        else:
            assert False

    @tensorflow_name_scoped
    def transform(self, x: tf.Tensor) -> tfd.Distribution:
        if self.parametrization is not None:
            x = self.parametrization.transform(x=x)

        if self.distribution == 'deterministic':
            loc = self.loc.transform(x=x)
            kwargs = dict(loc=loc)
        elif self.distribution == 'normal':
            loc = self.loc.transform(x=x)
            scale = tf.exp(x=self.scale.transform(x=x))
            kwargs = dict(loc=loc, scale=scale)

        return self.distribution_cls(validate_args=True, allow_nan_stats=False, **kwargs)
