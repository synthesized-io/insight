from collections import OrderedDict
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from .generative import Generative
from ..module import tensorflow_name_scoped
from ..values import Value


class VAE(Generative):

    def __init__(
        self, name: str, values: List[Value], distribution: str, latent_size : int, network: str,
        capacity: int, depth: int, batchnorm: bool, activation: str, optimizer: str,
        learning_rate: float, decay_steps: int, decay_rate: float, clip_gradients: float,
        beta: float, weight_decay: float
    ):
        super().__init__(name=name, values=values)

        self.latent_size = latent_size
        self.beta = beta

        # Total input and output size of all values
        input_size = 0
        output_size = 0
        for value in self.values:
            input_size += value.input_size()
            output_size += value.output_size()

        # Encoder: parametrized distribution
        parametrization = dict(
            module=network, layer_sizes=[capacity for _ in range(depth)], batchnorm=batchnorm,
            activation=activation, weight_decay=weight_decay
        )
        self.encoder = self.add_module(
            module='distribution', name='encoder', input_size=input_size, output_size=latent_size,
            distribution=distribution, parametrization=parametrization
        )

        # Decoder: parametrized distribution
        parametrization = dict(
            module=network, layer_sizes=[capacity for _ in range(depth)], batchnorm=batchnorm,
            activation=activation, weight_decay=weight_decay
        )
        self.decoder = self.add_module(
            module='distribution', name='decoder', input_size=self.encoder.size(),
            output_size=output_size, distribution='deterministic', parametrization=parametrization
        )

        # Optimizer
        self.optimizer = self.add_module(
            module='optimizer', name='optimizer', algorithm=optimizer, learning_rate=learning_rate,
            decay_steps=decay_steps, decay_rate=decay_rate, clip_gradients=clip_gradients
        )

    def specification(self) -> dict:
        spec = super().specification()
        spec.update(
            beta=self.beta, encoder=self.encoder.specification(),
            decoder=self.decoder.specification(), optimizer=self.optimizer.specification()
        )
        return spec

    def module_initialize(self) -> None:
        super().module_initialize()

        self.prior = tfd.Normal(
            loc=tf.zeros(shape=(self.latent_size)), scale=tf.ones(shape=(self.latent_size))
        )

    @tensorflow_name_scoped
    def learn(self, xs : Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Operation]:
        if len(xs) == 0:
            return tf.no_op()

        losses = OrderedDict()
        summaries = list()

        x = tf.concat(values=[xs[value.name] for value in self.values], axis=1)

        q = self.encoder.transform(x=x)
        assert q.reparameterization_type is tfd.FULLY_REPARAMETERIZED

        kldiv = tfd.kl_divergence(distribution_a=q, distribution_b=self.prior, allow_nan_stats=False)
        kldiv = tf.reduce_sum(input_tensor=kldiv, axis=1)
        kldiv = tf.reduce_mean(input_tensor=kldiv, axis=0)
        losses['kl-loss'] = self.beta * kldiv

        summaries.append(tf.contrib.summary.scalar(name='kl-divergence', tensor=kldiv))

        z = q.sample()

        p = self.decoder.transform(x=z)

        y = p.sample()

        ys = tf.split(
            value=y, num_or_size_splits=[value.output_size() for value in self.values], axis=1
        )

        for value, y in zip(self.values, ys):
            losses[value.name + '-loss'] = value.loss(x=y)

        reg_losses = tf.losses.get_regularization_losses()
        if len(reg_losses) > 0:
            losses['regularization-loss'] = tf.add_n(inputs=reg_losses)

        total_loss = tf.add_n(inputs=list(losses.values()))
        losses['total-loss'] = total_loss

        for name, loss in losses.items():
            summaries.append(tf.contrib.summary.scalar(name=name, tensor=loss))

        with tf.control_dependencies(control_inputs=summaries):
            optimized = self.optimizer.optimize(loss=loss, gradient_norms=True)

        return losses, optimized

    @tensorflow_name_scoped
    def synthesize(self, n : tf.Tensor) -> Dict[str, tf.Tensor]:
        z = self.prior.sample(sample_shape=(n,))

        p = self.decoder.transform(x=z)

        y = p.sample()

        ys = tf.split(
            value=y, num_or_size_splits=[value.output_size() for value in self.values], axis=1
        )

        synthesized = OrderedDict()
        for value, y in zip(self.values, ys):
            ys = value.output_tensors(x=y)
            for label, y in ys.items():
                synthesized[label] = y

        return synthesized
