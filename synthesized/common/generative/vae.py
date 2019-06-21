from collections import OrderedDict
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from .generative import Generative
from ..module import tensorflow_name_scoped
from ..values import Value


class VAE(Generative):
    """Variational auto-encoder.

    The VAE consists of an NN-parametrized input-conditioned encoder distribution q(z|x), a latent
    prior distribution p'(z), and an NN-parametrized latent-conditioned decoder distribution p(y|z).
    The optimized loss consists of the reconstruction loss per value, the KL loss, and the
    regularization loss. The input and output are concatenated / split tensors per value. The
    encoder and decoder network use the same hyperparameters.
    """

    def __init__(
        self, name: str, values: List[Value],
        # Latent distribution
        distribution: str, latent_size: int,
        # Encoder and decoder network
        network: str, capacity: int, depth: int, batchnorm: bool, activation: str,
        # Optimizer
        optimizer: str, learning_rate: float, decay_steps: int, decay_rate: float,
        clip_gradients: float,
        # Beta KL loss coefficient
        beta: float,
        # Weight decay
        weight_decay: float
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

        # Encoder: parametrized distribution q(z|x)
        parametrization = dict(
            module=network, layer_sizes=[capacity for _ in range(depth)], batchnorm=batchnorm,
            activation=activation, weight_decay=weight_decay
        )
        self.encoder = self.add_module(
            module='distribution', name='encoder', input_size=input_size, output_size=latent_size,
            distribution=distribution, parametrization=parametrization
        )

        # Decoder: parametrized distribution p(y|z)
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
            module='optimizer', name='optimizer', optimizer=optimizer, learning_rate=learning_rate,
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

        # Prior distribution p'(z)
        self.prior = tfd.Normal(
            loc=tf.zeros(shape=(self.latent_size,)), scale=tf.ones(shape=(self.latent_size,))
        )

    @tensorflow_name_scoped
    def learn(self, xs: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Operation]:
        """Training step for the generative model.

        Args:
            xs: An input tensor per value.

        Returns:
            A dictionary of loss tensors, and the optimization operation.

        """
        if len(xs) == 0:
            return dict(), tf.no_op()

        losses = OrderedDict()
        summaries = list()

        # Concatenate input tensors per value
        x = tf.concat(values=[xs[value.name] for value in self.values], axis=1)

        # Encoder q(z|x)
        q = self.encoder.parametrize(x=x)
        if q.reparameterization_type is not tfd.FULLY_REPARAMETERIZED:
            raise NotImplementedError

        # KL-divergence loss
        kldiv = tfd.kl_divergence(distribution_a=q, distribution_b=self.prior, allow_nan_stats=False)
        kldiv = tf.reduce_sum(input_tensor=kldiv, axis=1)
        kldiv = tf.reduce_mean(input_tensor=kldiv, axis=0)
        losses['kl-loss'] = self.beta * kldiv
        summaries.append(tf.contrib.summary.scalar(name='kl-divergence', tensor=kldiv))

        # Sample z ~ q(z|x)
        z = q.sample()

        # Decoder p(y|z)
        p = self.decoder.parametrize(x=z)

        # Sample y ~ p(y|z)
        y = p.sample()

        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.output_size() for value in self.values], axis=1
        )

        # Reconstruction loss per value
        for value, y in zip(self.values, ys):
            losses[value.name + '-loss'] = value.loss(x=y)

        # Regularization loss
        reg_losses = tf.losses.get_regularization_losses()
        if len(reg_losses) > 0:
            losses['regularization-loss'] = tf.add_n(inputs=reg_losses)

        # Total loss
        total_loss = tf.add_n(inputs=list(losses.values()))
        losses['total-loss'] = total_loss

        # Loss summaries
        for name, loss in losses.items():
            summaries.append(tf.contrib.summary.scalar(name=name, tensor=loss))

        # Make sure summary operations are executed
        with tf.control_dependencies(control_inputs=summaries):

            # Optimization step
            optimized = self.optimizer.optimize(
                loss=loss, summarize_gradient_norms=(self.summarizer is not None)
            )

        return losses, optimized

    @tensorflow_name_scoped
    def synthesize(self, n: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Generate the given number of instances.

        Args:
            n: The number of instances to generate.

        Returns:
            An output tensor per value.

        """
        # Sample z ~ p'(z)
        z = self.prior.sample(sample_shape=(n,))

        # Decoder p(y|z)
        p = self.decoder.parametrize(x=z)

        # Sample y ~ p(y|z)
        y = p.sample()

        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.output_size() for value in self.values], axis=1
        )

        # Output tensors per value
        synthesized = OrderedDict()
        for value, y in zip(self.values, ys):
            ys = value.output_tensors(x=y)
            for label, y in ys.items():
                synthesized[label] = y

        return synthesized
